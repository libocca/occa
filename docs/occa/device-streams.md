Streams
========================================

Streams are another concept exposed in OCCA which are found in the GPU programming model.
The role of a stream is to store commands and execute them.
Similar to how our computers run hundreds of applications through resource scheduling, multiple streams could run through context switching.

OCCA device objects are able to create streams but contain only one active stream.
The active or current stream in the device handles consequent kernel calls and memory transfers.
This does not guarantee the device will execute those commands first, it only guarantees the commands execute after all other unfinished commands already in the stream.
