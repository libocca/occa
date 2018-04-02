# Device Streams

Streams are another concept exposed in OCCA which are found in the GPU programming model.

The role of a stream is to store commands and execute them.
Similar to how our computers run hundreds of applications through resource scheduling, multiple streams could run through context switching.

OCCA device objects are able to create multiple streams but keep only one active stream;
The active stream in the device handles kernel calls and memory transfers until the device's active stream is changed.

By default, kernel calls and memory transfers in a device will execute in-order.
However, this does not guarantee in-order completion across streams.

!> TODO: Missing API Section
