#include <memory>

#include "Judy.h"

#if 0

static std::allocator<Word_t> judy_allocator;

extern "C" {
    
Word_t JudyMalloc(
	Word_t Words)
{
    return (Word_t)judy_allocator.allocate(Words);
} // JudyMalloc()


// ****************************************************************************
// J U D Y   F R E E

void JudyFree(
	void * PWord,
	Word_t Words)
{
    return judy_allocator.deallocate((Word_t *)PWord, Words);
} // JudyFree()


// ****************************************************************************
// J U D Y   M A L L O C
//
// Higher-level "wrapper" for allocating objects that need not be in RAM,
// although at this time they are in fact only in RAM.  Later we hope that some
// entire subtrees (at a JPM or branch) can be "virtual", so their allocations
// and frees should go through this level.

Word_t JudyMallocVirtual(
	Word_t Words)
{
	return(JudyMalloc(Words));

} // JudyMallocVirtual()


// ****************************************************************************
// J U D Y   F R E E

void JudyFreeVirtual(
	void * PWord,
	Word_t Words)
{
        JudyFree(PWord, Words);

} // JudyFreeVirtual()

} // extern "C"

#endif


