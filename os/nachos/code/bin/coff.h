/* coff.h
 *   Data structures that describe the MIPS COFF format.
 */

struct filehdr {
        unsigned short  f_magic;        /* magic number */
        unsigned short  f_nscns;        /* number of sections */
        int             f_timdat;       /* time & date stamp */
        int             f_symptr;       /* file pointer to symbolic header */
        int             f_nsyms;        /* sizeof(symbolic hdr) */
        unsigned short  f_opthdr;       /* sizeof(optional hdr) */
        unsigned short  f_flags;        /* flags */
      };

#define  MIPSELMAGIC    0x0162

#define OMAGIC  0407
#define SOMAGIC 0x0701

typedef struct aouthdr {
        short   magic;          /* see above                            */
        short   vstamp;         /* version stamp                        */
        int     tsize;          /* text size in bytes, padded to DW bdry*/
        int     dsize;          /* initialized data "  "                */
        int     bsize;          /* uninitialized data "   "             */
        int     entry;          /* entry pt.                            */
        int     text_start;     /* base of text used for this file      */
        int     data_start;     /* base of data used for this file      */
        int     bss_start;      /* base of bss used for this file       */
        int     gprmask;        /* general purpose register mask        */
        int     cprmask[4];     /* co-processor register masks          */
        int     gp_value;       /* the gp value used for this object    */
      } AOUTHDR;
#define AOUTHSZ sizeof(AOUTHDR)
 

struct scnhdr {
        char            s_name[8];      /* section name */
        int             s_paddr;        /* physical address, aliased s_nlib */
        int             s_vaddr;        /* virtual address */
        int             s_size;         /* section size */
        int             s_scnptr;       /* file ptr to raw data for section */
        int             s_relptr;       /* file ptr to relocation */
        int             s_lnnoptr;      /* file ptr to gp histogram */
        unsigned short  s_nreloc;       /* number of relocation entries */
        unsigned short  s_nlnno;        /* number of gp histogram entries */
        int             s_flags;        /* flags */
      };
 
