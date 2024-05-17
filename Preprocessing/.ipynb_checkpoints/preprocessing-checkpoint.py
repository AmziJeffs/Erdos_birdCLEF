# Preprocessing functions for working with audio


# Takes inputs
#     - seq, a Collection (i.e. a sized iterable)
#     - window_size, an integer
#     - stride, an integer
#     - align_left, a Boolean
# Returns a list of segments in seq, each with length window_size,
# consecutive segments offside by stride. If align_left is False,
# these are shift right so that the last slice is "flush" with the 
# righthand side of seq.
# Our use case is when seq is an audio clip. 
def slices(seq, window_size, stride = None, align_left = True):
    # If one window covers the whole collection, just return the collection
    if window_size >= len(seq):
        return [seq]

    indices = [i for i in range(len(seq))]
    index_slices = []
    left_pointer = 0
    while left_pointer + window_size < len(seq):
        index_slices += [[left_pointer, left_pointer + window_size]]
        left_pointer += stride

    if align_left == False:
        offset = len(seq)-(left_pointer-stride)-window_size
        index_slices = [[a+offset, b+offset] for [a,b] in index_slices]

    return [seq[a:b] for [a,b] in index_slices]

    

