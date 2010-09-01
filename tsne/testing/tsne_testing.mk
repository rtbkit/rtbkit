$(eval $(call test,tsne_test,tsne utils arch,boost timed))
$(eval $(call pytest,tsne_python_test,tsne,manual))
