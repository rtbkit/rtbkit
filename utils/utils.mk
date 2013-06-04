$(eval $(call library,variadic_hash,variadic_hash.cc,cityhash))

$(eval $(call include_sub_make,testing))

