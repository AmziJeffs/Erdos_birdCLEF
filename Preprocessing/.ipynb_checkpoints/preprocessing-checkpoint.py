# Preprocessing functions for working with audio


# Takes inputs
#     - seq, a Collection (i.e. a sized iterable)
#     - window_size, an integer
#     - stride, an integer
#     - align_left, a Boolean
#     - return_scraps, a Boolean
# Returns a list of segments in seq, each with length window_size,
# consecutive segments offside by stride. If align_left is False,
# these are shift right so that the last slice is "flush" with the 
# righthand side of seq.
# If return_scraps is True, and we could have fit a partial window 
# over the rest of our sequence, we return that partial window. If 
# align_left is False, the "scrap" comes from the beginning of seq.
# Our use case is when seq is an audio clip. For this reason, we default
# the window size to 160000 = 5 seconds of samples at 32000 hz
def slices(seq, window_size = 160000, stride = None, align_left = True, return_scraps = False):
    
    # If one window is larger than the sequence, just return the scraps or nothing
    if window_size > len(seq):
        if return_scraps == True:
            return [seq]
        else:
            return []

    # If stride is None, it defaults to window_size
    if stride == None:
        stride = window_size
    
    index_slices = []
    left_pointer = 0
    while left_pointer + window_size <= len(seq):
        index_slices += [[left_pointer, left_pointer + window_size]]
        left_pointer += stride

    if align_left == False:
        offset = len(seq)-(left_pointer-stride)-window_size
        index_slices = [[a+offset, b+offset] for [a,b] in index_slices]

    if return_scraps == True and left_pointer + window_size > len(seq):
        if align_left == True:
            index_slices += [[left_pointer, len(seq)]]
        else:
            index_slices += [[0, len(seq) - left_pointer]]
    
    return [seq[a:b] for [a,b] in index_slices]

    

