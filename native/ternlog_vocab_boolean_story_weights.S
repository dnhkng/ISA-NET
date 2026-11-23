// Packed AX1 vocabulary Boolean parameters.
#ifndef TERNLOG_BOOLEAN_WEIGHTS_PATH
#define TERNLOG_BOOLEAN_WEIGHTS_PATH "native/ternlog_vocab_boolean_story_weights.bin"
#endif
.section __DATA,__const,regular
.balign 64
.globl _ternlog_vocab_story_weights
_ternlog_vocab_story_weights:
.incbin TERNLOG_BOOLEAN_WEIGHTS_PATH
.globl _ternlog_vocab_story_weights_end
_ternlog_vocab_story_weights_end:
