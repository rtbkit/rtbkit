# docker.mk
# Jeremy Barnes, 22 December 2013
# Copyright (c) 2013 Datacratic Inc and Jeremy Barnes.
# Available under the Apache license 2.0.

# This file contains support for using docker (http://docker.io) with
# jml-build.  This allows docker images to be created directly with make,
# that can then be deployed on production machines.


###############################################################################
# External variables:
#
# DOCKER_REGISTRY: the docker registry used for packages.  By default it's
# empty, which means that the public dotcloud registry will be used.  Be
# careful with this; if you push your images they will be made public.
#
# This should be overridden in local.mk, not edited here.

DOCKER_REGISTRY?=

# DOCKER_USER: the docker user ID for accessing repositories.
#
# This should be overridden in local.mk, not edited here.

DOCKER_USER?=$(shell whoami)

# DOCKER_BASE_IMAGE: the docker image that docker images will be built on top
# of.  Note that you can override this per target.  For example,
#
# docker_my_make_target: DOCKER_BASE_IMAGE=another_image_name
#
# This should be overridden in local.mk, not edited here.

DOCKER_BASE_IMAGE?=$(DOCKER_REGISTRY)$(DOCKER_USER)dependencies

# DOCKER_GET_REVISION_SCRIPT: this is the script used to get the revision
# ID for the docker image.
#
# The script takes one single command line argument, the make target, and
# should print on stdout the revision ID.  If it returns 0 then the build
# can continue; if it returns 1 then the build will not be allowed unless
# DOCKER_ALLOW_DIRTY is set.
#
# This should be overridden in local.mk, not edited here.

DOCKER_GET_REVISION_SCRIPT?=$(JML_BUILD)/get_git_revision.sh 

# DOCKER_ALLOW_DIRTY: if this is defined, then a docker container will be
# allowed to be built, even if the git tree is dirty (in other words, even
# if the build could not be reproduced).
#
# This should be added to local.mk, not edited here.

#DOCKER_ALLOW_DIRTY:=1

# DOCKER_PUSH: if this is defined, then docker will be asked to push the
# container to the repository after it has been successfully built.
#
# This should be added to local.mk, not edited here.

#DOCKER_PUSH:=1

# DOCKER_GLOBAL_DEPS: these are a set of make targets that must be run
# before any docker installation.

# DOCKER_TARGET_DEPS: Anything in this variable (which should be overridden
# on a per-target basis) will be made before the docker image.

# DOCKER_TAG: if this is defined, the given tag will also be applied to the
# built image.  By default it's "latest" which is expected by most docker
# tooling, but can be changed to something else or undefined if required.

# DOCKER_COMMIT_ARGS: Anything in this variable (which should be overridden
# on a per-target basis) will be passed to docker commit as arguments.

# DOCKER_POST_INSTALL_SCRIPT: If this variable is set (it should be set on a
# per-target basis) then the given script will be run inside the docker
# container after the container is created.  It can be used to modify the
# container before it is committed.

DOCKER_TAG:=latest



# Docker target (generic).  If you make docker_target_name, it will make
# target_name and install it inside a docker image.
#
# In order to determine the tag, by default this rule will call the
# get_git_revision script that will return a revision ID from git to tag
# the image with.  The script will also ensure that everything used in the
# build is checked in so that the build is reproducible.


#docker_%:	$(TMPBIN)/%.iid

docker_%: % $(DOCKER_GLOBAL_DEPS) $(DOCKER_TARGET_DEPS)
	@BUILD=$(BUILD) bash $(DOCKER_GET_REVISION_SCRIPT) $(<) > $(TMPBIN)/$(<).rid $(if $(DOCKER_ALLOW_DIRTY), || true,)
	echo "revision" `cat $(TMPBIN)/$(<).rid`
	@echo "Building $(<) for use within docker"
	+make TMPBIN=$(TMPBIN) LIB=$(TMPBIN)/docker-$(<)/lib BIN=$(TMPBIN)/docker-$(<)/bin ETC=$(TMPBIN)/docker-$(<)/etc $(<)
	@echo "Creating container"
	@rm -f $(TMPBIN)/$(<).cid
	docker run -cidfile $(TMPBIN)/$(<).cid -v `pwd`:/tmp/build $(DOCKER_BASE_IMAGE) sh /tmp/build/$(JML_BUILD)/docker_install_inside_container.sh /tmp/build/$(TMPBIN)/docker-$(<) $(if $(DOCKER_POST_INSTALL_SCRIPT),/tmp/build/$(DOCKER_POST_INSTALL_SCRIPT))
	cat $(TMPBIN)/$(<).cid
	echo docker commit `cat $(TMPBIN)/$(<).cid` $(DOCKER_REGISTRY)$(DOCKER_USER)$(<):`cat $(TMPBIN)/$(<).rid`
	docker commit $(DOCKER_COMMIT_ARGS) `cat $(TMPBIN)/$(<).cid` $(DOCKER_REGISTRY)$(DOCKER_USER)$(<):`cat $(TMPBIN)/$(<).rid` > $(TMPBIN)/$<.iid && cat $(TMPBIN)/$<.iid
	$(if $(DOCKER_TAG),docker tag `cat $(TMPBIN)/$(<).iid` $(DOCKER_REGISTRY)$(DOCKER_USER)$(<):$(DOCKER_TAG))
	@docker rm `cat $(TMPBIN)/$(<).cid`
	$(if $(DOCKER_PUSH),docker push $(DOCKER_REGISTRY)$(DOCKER_USER)$(<))
	@echo $(COLOR_WHITE)Created $(if $(DOCKER_PUSH),and pushed )$(COLOR_BOLD)$(DOCKER_REGISTRY)$(DOCKER_USER)$(<):`cat $(TMPBIN)/$(<).rid`$(COLOR_RESET) as image $(COLOR_WHITE)$(COLOR_BOLD)`cat $(TMPBIN)/$<.iid`$(COLOR_RESET)
