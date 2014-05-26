/* pipeline.cc
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

std::string Environment::expandVariables(std::string const & text) const {
    size_t i = 0;
    std::string result;
    for(;;) {
        auto j = text.find("%{", i);
        result.append(text, i, j - i);

        if(j == std::string::npos) {
            break;
        }

        i = j + 2;
        j = text.find("}", i);

        if(j == std::string::npos) {
            break;
        }

        std::string key(text, i, j - i);
        result += getVariable(key);

        i = j + 1;
    }

    return result;
}

std::string Environment::getVariable(std::string const & text) const {
    auto k = keys.find(text);
    return keys.end() != k ? k->second : text;
}

void Environment::set(std::string key, std::string value) {
    keys[key] = value;
}

EnvironmentDescription::
EnvironmentDescription() {
}

Pipeline::Pipeline() :
    Blocks(this),
    environment(this, "environment") {
}

