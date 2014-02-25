#!/usr/bin/env python
#-------------------------------------------------------------------------------#
# A tool which exports a development tree to userland.
# The development tree is supposed to properly built and tested.
#-------------------------------------------------------------------------------#
import re
import os 
import bz2 
import shutil
import os.path
import tarfile
import logging
import argparse


LOGGERS_CREATED = set()
LOGGER_LEVEL = logging.DEBUG

# --- Sets up a console logger
def setup_console_logger(name):
    logger = logging.getLogger(name)

    if name not in LOGGERS_CREATED:
        logger.setLevel(LOGGER_LEVEL)
        ch = logging.StreamHandler()
        ch.setLevel(LOGGER_LEVEL)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - '
                                          '%(levelname)s - %(message)s'))
        logger.addHandler(ch)
        LOGGERS_CREATED.add(name)

    return logger


# -- Reading the arguments
def read_arguments():
    parser = argparse.ArgumentParser(description='tool for exporting '
                                      'an RTBkit tree')

    parser.add_argument('--rtbkit_root', required=True,
        help='path to the rtbkit root directory')

    parser.add_argument('--local_root', required=True,
        help='path to the local directory where platform-deps were installed')

    parser.add_argument('--prefix',
        help='Optionally specify a basename for the export. default to rtbkit')

    return parser.parse_args()

    
def main(args):
    cwd    = os.getcwd ()
    logger = setup_console_logger('export_rtbkit')
    
    # rtbkit_root = '.'
    # local_root = '../local'
    bin_root = args.rtbkit_root + '/build/x86_64/bin'
    base = 'rtbkit'
    if args.prefix: base = args.prefix
    tar_filename=base + '.tar'

    tar_out = tarfile.open(name=tar_filename ,mode='w', dereference=True)
    
    logger.info ('opening archive: ' + tar_filename)
    def tarfile_add (arc, name, arcname):
        logger.info ('adding %s to %s'%(name, arcname))
        arc.add(name=name,arcname=arcname)

    def strip_root(root, path):
        root_vec = root.split('/')
        path_vec = path.split('/')
        if len(root_vec) >= len(path_vec): return path
        v = '/'.join(path_vec[len(root_vec):])
        return v
    
    def is_header_file (path):
	# print path
        if (path.endswith('.h') or path.endswith('.hh') or path.endswith('.hpp') or path.endswith('.h.in')):
            return True
        return False
    
    # we do do rtbkit includes
    exclude_re = re.compile ('.*\/(build|examples)\/.*')
    for root, dirs, files in os.walk(args.rtbkit_root):
        if '.git' in dirs:
    	    dirs.remove ('.git')
        for f in files:
            p= os.path.join(root,f)
    	    if is_header_file(p) and not exclude_re.match(p):
    	        nname = base+'/include/rtbkit/'+strip_root(args.rtbkit_root,p)
    	        tarfile_add(tar_out,p,nname)
    
    # we now deal with rtbkit binaries
    exclude_re = re.compile('.*\.(mk|[a-z0-9]{19,}\.so)$')
    for root, dirs, files in os.walk(bin_root):
        if '.git' in dirs:
    	    dirs.remove ('.git')
        for f in files:
            p= os.path.join(root,f)
    	    if f[0] != '.' and not exclude_re.match(p): 
    	        rep = '/lib/' if p.endswith('.so') else '/bin/'
    	        tarfile_add(tar_out, p,base+rep+f)
    	        pass
    
    # we now add the examples directory
    tarfile_add (tar_out, args.rtbkit_root+'/examples', base+'/examples')
    
    # we deal with local includes
    tarfile_add (tar_out, args.local_root+'/include',base+'/include') 
    
    # we now deal with local libs
    tarfile_add (tar_out, args.local_root+'/lib',base+'/lib') 
    
    # we now deal with local libs
    tarfile_add (tar_out, args.local_root+'/bin',base+'/bin') 
    
    
    # grab our sample json configs
    sample_configs= ['sample.bootstrap.json','sample.launch.json','sample.launch.scale.json','sample.redis.conf','sample.zookeeper.conf']
    for sample_config in sample_configs:
        fn=args.rtbkit_root+'/'+sample_config
        if os.path.isfile(fn):
            tarfile_add (tar_out, fn, base+'/config/'+sample_config)
        else:
            logger.warning ('missing %s'%sample_config)
	    continue
    	if sample_config=='sample.zookeeper.conf': 
                tarfile_add (tar_out, fn, base+'/bin/zookeeper/bin/'+sample_config)
    
    logger.info ('closing archive')
    tar_out.close()



    # this hack alows to work around
    # various issues related to hard coded
    # paths etc.
    # we need to re-open the tarbal setting dereference 
    # to False (i.e., symlinks will be copied)
    logger.info ('re-opening archive with dereference=False')
    tar_out = tarfile.open(name=tar_filename ,mode='a', dereference=False)
    tarfile_add (tar_out, args.rtbkit_root+'/rtbkit', base+'/rtbkit')
    shutil.rmtree('/tmp/xx/build',True) 
    os.makedirs('/tmp/xx/build/x86_64')
    os.chdir('/tmp/xx/build/x86_64')
    os.symlink('../../bin','bin')
    tarfile_add (tar_out, '/tmp/xx/build', base+'/build')
    tar_out.close()
    logger.info ('closing archive')

    logger.info('starting compressing')
    os.chdir (cwd)
    from shutil import copyfileobj
    with open(tar_filename, 'rb') as input:
        with bz2.BZ2File(tar_filename+'.bz2', 'wb', compresslevel=9) as output:
	     copyfileobj(input, output)
    logger.info('compressing done')



##########################
if __name__ == '__main__':
    args = read_arguments()
    main(args)

