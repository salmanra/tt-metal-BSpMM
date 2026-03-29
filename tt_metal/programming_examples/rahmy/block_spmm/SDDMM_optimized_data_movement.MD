# How we can use a modified version of CDA to optimize data movement for SDDMM

I think I actually have a SDDMM optimization. 
Think of the sparse matrix as “compact”. You can analyze the sharing pattern by labeling nonzero blocks of the sparse matrix by their column index, then “compacting” the sparse matrix, then within a single column of the compact matrix, identical original column indices represent groups of cores which will share data
Then cut up each row of the packed matrix which has more than num_cores_x nonzero blocks. Then “pack” all the rows which have less than num_cores_x nonzero blocks together into groups which do not exceed num_cores_x nonzeros. Then disrespect the rows, and pack every group with blocks until there is only one group with fewer than num_cores_x nonzero blocks.  
Congratulations, you now have three kinds of groups:

num_cores_x blocks from the same row
num_cores_x blocks from different rows
No more than one group of < num_cores_x blocks from 1 or more rows

Then iterate over the core grid until every group has been assigned a row of cores. Many compile time and runtime args must be passed to each core this way. 

Unlike SpMM, the two data movement kernels on a Tensix tile for SDDMM will have symmetric data movement schemes. First, they will ierate over the number of output blocks that Tensix tile has been assigned. Then they will grab their indexing data form reading a block of their input matrices as a runtime argument, then discover the set of cores that they will be sharing data with. Descriptions for how each core will discover their share set can be found as a many-line comment in the existing data movement kernels for sddmm which contain the substring "SnF". Then, with the role and share set computed, each core will perform CDA to obtain their input blocks across the entire reduction dimension. In SDDMM, the reduction dimension is fixed, so we only have to compute the share set associated with an output block once. Once We have read all of the input blocks for across the reduction dimension, we perform the barrier synchronization associated with CDA. For the SDDMM data movement kernel doing the writeback, we do this synchronization after the writeback. 