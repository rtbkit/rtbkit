/* fs_utils.cc
   Wolfgang Sourdeau, February 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A set of file-system abstraction functions intended to support common
   operations among different fs types or alikes.
*/

#include <libgen.h>

#include <memory>
#include <map>
#include <mutex>

#include "boost/filesystem.hpp"
#include "googleurl/src/url_util.h"

#include "fs_utils.h"
#include "jml/utils/guard.h"
#include "jml/utils/file_functions.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


using namespace std;
using namespace Datacratic;
namespace fs = boost::filesystem;

namespace {

/* registry */

struct Registry 
{
    std::mutex mutex;
    map<string, std::unique_ptr<const UrlFsHandler> > handlers;
};

Registry& getRegistry()
{
    static Registry* registry = new Registry;
    return *registry;
}

} // file scope

namespace Datacratic {

/* LOCALURLFSHANDLER */

static FsObjectInfo extractInfo(const struct stat & stats)
{
    FsObjectInfo objectInfo;

    objectInfo.exists = true;
    objectInfo.lastModified = Date::fromTimespec(stats.st_mtim);
    objectInfo.size = stats.st_size;

    return objectInfo;
}

struct LocalUrlFsHandler : public UrlFsHandler {

    virtual FsObjectInfo getInfo(const Url & url) const
    {
        struct stat stats;
        string path = url.path();

        // cerr << "fs info on path: " + path + "\n";
        int res = ::stat(path.c_str(), &stats);
        if (res == -1) {
            if (errno == ENOENT) {
                return FsObjectInfo();
            }
            throw ML::Exception(errno, "stat");
        }

        // TODO: owner ID (uid) and name (uname)

        return extractInfo(stats);
    }

    virtual FsObjectInfo tryGetInfo(const Url & url) const
    {
        struct stat stats;
        string path = url.path();

        // cerr << "fs info on path: " + path + "\n";
        int res = ::stat(path.c_str(), &stats);
        if (res == -1) {
            return FsObjectInfo();
        }

        return extractInfo(stats);
    }
    
    virtual void makeDirectory(const Url & url) const
    {
        boost::system::error_code ec;
        string path = url.path();
        if (!fs::exists(path) && !fs::create_directories(path, ec)) {
            throw ML::Exception(ec.message());
        }
    }

    virtual bool erase(const Url & url, bool throwException) const
    {
        string path = url.path();
        int res = ::unlink(path.c_str());
        if (res == -1) {
            if (throwException) {
                throw ML::Exception(errno, "unlink");
            }
            else return false;
        }
        return true;
    }

    virtual bool forEach(const Url & prefix,
                         const OnUriObject & onObject,
                         const OnUriSubdir & onSubdir,
                         const std::string & delimiter,
                         const std::string & startAt) const
    {
        using namespace ML;

        if (startAt != "")
            throw ML::Exception("not implemented: startAt for local files");
        if (delimiter != "/")
            throw ML::Exception("not implemented: delimiters other than '/' "
                                "for local files");
        

        bool result = true;
        auto onFileFound = [&] (const std::string & dir,
                                const std::string & basename,
                                const struct stat & stats,
                                FileType type,
                                int depth) -> ML::FileAction
            {
                if (type == FT_FILE) {
                    result = onObject(dir + "/" + basename,
                                      extractInfo(stats),
                                      depth);
                    if (!result)
                        return FA_STOP;
                    else return FA_CONTINUE;
                }
                else if (type == FT_DIR) {
                    if (!onSubdir)
                        return FA_CONTINUE;
                    else if (onSubdir(dir + "/" + basename,
                                      depth))
                        return FA_CONTINUE;
                    else return FA_SKIP_SUBTREE;
                }
                else return FA_CONTINUE;
            };

        scanFiles(prefix.path(), onFileFound, -1);

        return result;
    }
};


const UrlFsHandler * findFsHandler(const string & scheme)
{
    auto& registry = getRegistry();

    std::unique_lock<std::mutex> guard(registry.mutex);
    auto handler = registry.handlers.find(scheme);
    if (handler == registry.handlers.end()) {
        throw ML::Exception("no handler found for scheme: " + scheme);
    }
    return handler->second.get();
}


struct AtInit {
    AtInit() {
        registerUrlFsHandler("file", new LocalUrlFsHandler());
    }
} atInit;


/* ensures that local filenames are represented as urls */
Url makeUrl(const string & urlStr)
{
    if (urlStr.empty())
        throw ML::Exception("can't makeUrl on empty url");

    /* scheme is specified */
    if (urlStr.find("://") != string::npos) {
        return Url(urlStr);
    }
    /* absolute local filenames */
    else if (urlStr[0] == '/') {
        return Url("file://" + urlStr);
    }
    /* relative filenames */
    else {
        char cCurDir[PATH_MAX + 1];
        string filename(getcwd(cCurDir, sizeof(cCurDir)));
        filename += "/" + urlStr;

        return Url("file://" + filename);
    }
}

}


namespace Datacratic {

/* URLFSHANDLER */

size_t
UrlFsHandler::
getSize(const Url & url) const
{
    return getInfo(url).size;
}

string
UrlFsHandler::
getEtag(const Url & url) const
{
    return getInfo(url).etag;
}


/* registry */

void registerUrlFsHandler(const std::string & scheme,
                          UrlFsHandler * handler)
{
    auto& registry = getRegistry();

    if (registry.handlers.find(scheme) != registry.handlers.end()) {
        throw ML::Exception("fs handler already registered");
    }

    /* this enables googleuri to parse our urls properly */
    url_util::AddStandardScheme(scheme.c_str());

    registry.handlers[scheme].reset(handler);
}

FsObjectInfo
tryGetUriObjectInfo(const std::string & url)
{
    Url realUrl = makeUrl(url);
    return findFsHandler(realUrl.scheme())->tryGetInfo(realUrl);
}

FsObjectInfo
getUriObjectInfo(const std::string & url)
{
    Url realUrl = makeUrl(url);
    return findFsHandler(realUrl.scheme())->getInfo(realUrl);
}
 
size_t
getUriSize(const std::string & url)
{
    Url realUrl = makeUrl(url);
    return findFsHandler(realUrl.scheme())->getSize(realUrl);
}

std::string
getUriEtag(const std::string & url)
{
    Url realUrl = makeUrl(url);
    return findFsHandler(realUrl.scheme())->getEtag(realUrl);
}

void
makeUriDirectory(const std::string & url)
{
    string dirUrl(url);
    size_t slashIdx = dirUrl.rfind('/');
    if (slashIdx == string::npos) {
        throw ML::Exception("makeUriDirectory cannot work on filenames: instead of " + url + " you should probably write file://" + url);
    }
    dirUrl.resize(slashIdx);

    // cerr << "url: " + url + "/dirUrl: " + dirUrl + "\n";

    Url realUrl = makeUrl(dirUrl);
    findFsHandler(realUrl.scheme())->makeDirectory(realUrl);
}

bool
eraseUriObject(const std::string & url, bool throwException)
{
    Url realUrl = makeUrl(url);
    return findFsHandler(realUrl.scheme())->erase(realUrl, throwException);
}

bool
tryEraseUriObject(const std::string & uri)
{
    return eraseUriObject(uri, false);
}

bool forEachUriObject(const std::string & urlPrefix,
                      const OnUriObject & onObject,
                      const OnUriSubdir & onSubdir,
                      const std::string & delimiter,
                      const std::string & startAt)
{
    Url realUrl = makeUrl(urlPrefix);
    return findFsHandler(realUrl.scheme())
        ->forEach(realUrl, onObject, onSubdir, delimiter, startAt);
}

string
baseName(const std::string & filename)
{
    char *fnCopy = ::strdup(filename.c_str());
    ML::Call_Guard guard([&] {
        ::free(fnCopy);
    });
    char *dirNameC = ::basename(fnCopy);
    string dirname(dirNameC);

    return dirname;
}

string
dirName(const std::string & filename)
{
    char *fnCopy = ::strdup(filename.c_str());
    ML::Call_Guard guard([&] {
        ::free(fnCopy);
    });
    char *dirNameC = ::dirname(fnCopy);
    string dirname(dirNameC);

    return dirname;
}

} // namespace Datacratic
