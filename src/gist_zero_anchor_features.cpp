// GIST1M uses float32 TexMex vectors, so the generic float32 zero-anchor
// collector is shared with Deep100M.  Keeping this as a separate target makes
// dataset-specific experiment commands and artifacts unambiguous.
#include "deep100m_zero_anchor_features.cpp"
