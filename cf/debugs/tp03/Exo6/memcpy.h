typedef unsigned int size_t;
/* Informal  specification:
’src’ is an array of size ’src_size’
’dest’ is an array of size ’dest_size’
The function copies ’size’ items from array ’src’ into array ’dest’,
with offsets ’src_offset’ and ’dest_offset’.
Returns -1 in case of error (i.e., if size > (src_size - src_offset), if src_offset >= src_size, and similarly for dest).
*/

/**
 * @brief Copies the content of src[src_offset .. src_offset+size] into dest[dest_offset .. dest_offset+size]. Return 0 in case of success and -1 if a failure occurs. In that last case, the content of dest is left unchanged.
 * 
 * @param src an array to be read from
 * @param src_size the size of @p src
 * @param src_offset the starting position read in @p src
 * @param dest an array to be written to
 * @param dest_size the size of @p dest
 * @param dest_offset the starting position written to in @p dest
 * @param size the number of values to copy
 * @return int 0 in case of success, -1 in case of failure.
 * @pre @p src and @p dest must be valid arrays of their respective size
 * @pre @p src and @p dest must be disjoint arrays
 * 
 */
int memcpy(char *src, size_t src_size, size_t src_offset, char *dest, size_t dest_size, size_t dest_offset, size_t size);