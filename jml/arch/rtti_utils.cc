/* rtti_utils.cc
   Jeremy Barnes, 18 October 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.
   Copyright (c) 2010 Datacratic.  All rights reserved.

*/

#include "rtti_utils.h"
#include <cxxabi.h>
#include <iostream>
#include "demangle.h"

using namespace abi;
using namespace std;

namespace __cxxabiv1 {

struct __class_type_info::__upcast_result
{
    const void *dst_ptr;        // pointer to caught object
    __sub_kind part2dst;        // path from current base to target
    int src_details;            // hints about the source type heirarchy
    const __class_type_info *base_type; // where we found the target,
                                // if in vbase the __class_type_info of vbase
                                // if a non-virtual base then 1
                                // else NULL
public:
    __upcast_result (int d)
        :dst_ptr (NULL), part2dst (__unknown), src_details (d), base_type (NULL)
    {}
};

} // namespace abi

namespace ML {

const void * is_convertible(const std::type_info & from_type,
                            const std::type_info & to_type,
                            const void * obj)
{
    const abi::__class_type_info * fromcti
        = dynamic_cast<const abi::__class_type_info *>(&from_type);
    const abi::__class_type_info * tocti
        = dynamic_cast<const abi::__class_type_info *>(&to_type);

    //cerr << "converting " << demangle(from_type) << " to " << demangle(to_typ e)
    //     << endl;

    if (fromcti && tocti) {
        abi::__class_type_info::__upcast_result
            ur( __vmi_class_type_info::__flags_unknown_mask);
        
        bool could_upcast = fromcti->__do_upcast(tocti, obj, ur);

#if 0
        cerr << "ur.dst_ptr = " << ur.dst_ptr << endl;
        cerr << "part2dst = " << ur.part2dst << endl;
        cerr << "src_details = " << ur.src_details << endl;
        cerr << "base_type = " << ur.base_type << endl;

        cerr << "  unknown = " << __class_type_info::__unknown << endl;
        cerr << "  not_contained = " << __class_type_info::__not_contained << endl;
        cerr << "  contained_ambig = " << __class_type_info::__contained_ambig
             << endl;
        cerr << "  contained_virtual_mask = " << __class_type_info::__contained_virtual_mask
             << endl;
        cerr << "  contained_public_mask = " << __class_type_info::__contained_public_mask
             << endl;
        cerr << "  contained_mask = " << __class_type_info::__contained_mask
             << endl;
#endif

        if (could_upcast) {
            //cerr << "could upcast" << endl;
            return ur.dst_ptr;
        }
        
        //cerr << "couldn't upcast" << endl;
        
        //const char * adj_ptr = (const char *)obj;
        //bool result = to_type.__do_catch(&from_type, (void **)&adj_ptr, 0);
        ptrdiff_t src2dst = -1;
        //if (result) src2dst = adj_ptr - (const char *)obj;

        void * res = abi::__dynamic_cast(obj, fromcti, tocti, src2dst);
        return res;
    }

    const char * adj_ptr = (const char *)obj;
    bool result = to_type.__do_catch(&from_type, (void **)&adj_ptr, 0);
    return (result ? adj_ptr : 0);
}

} // namespace ML
   
