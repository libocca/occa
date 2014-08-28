using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using liboccaSharp;
using System.IO;

namespace addVectors_sharp {
    class Program {
        static void Main(string[] args) {
            liboccaSharp.Setup.SetDllSearchPath(ReleaseOrDebug: false);

            int entries = 5;

            float[] a  = new float[entries];
            float[] b  = new float[entries];
            float[] ab = new float[entries];

            for(int i = 0; i < entries; ++i) {
                a[i] = (float)i;
                b[i] = (float)(1 - i);
                ab[i] = 0;
            }

            int int_size = sizeof(int);
            int pointer_size = IntPtr.Size;
            Console.WriteLine("Hello from addVectors: "
                + " integer size: " + int_size
                + " pointer size: " + pointer_size);


            Mode mode = Mode.OpenMP;

            int platformID = 0;
            int deviceID   = 0;

            liboccaSharp.Device device;
            liboccaSharp.Kernel addVectors;
            liboccaSharp.Memory o_a, o_b, o_ab;

            device = new Device(mode, platformID, deviceID);

            o_a = device.malloc(entries * sizeof(float));
            o_b = device.malloc(entries * sizeof(float));
            o_ab = device.malloc(entries * sizeof(float));

            string occaDir = System.Environment.GetEnvironmentVariable("OCCA_DIR");
            string addVectors_occa = "addVectors.occa";
            if(occaDir != null && occaDir.Length > 0) {

                addVectors_occa = Path.Combine(occaDir, "examples", "addVectors", addVectors_occa);
            }

            addVectors = device.buildKernelFromSource(addVectors_occa, "addVectors");

            addVectors.Dims = 1;
            addVectors.itemsDim.xi = 2; // int itemsPerGroup = 2;
            addVectors.groupsDim.xi = ((entries + addVectors.itemsDim.xi - 1) / addVectors.itemsDim.xi);
            
            
            o_a.copyFrom(a);
            o_b.copyFrom(b);

            addVectors.Invoke(entries, o_a, o_b, o_ab);

            o_ab.copyTo(ab);

            for(int i = 0; i < 5; ++i)
                Console.WriteLine(i + ": " + ab[i]);


            addVectors.Dispose();
            o_a.Dispose();
            o_b.Dispose();
            o_ab.Dispose();
            device.Dispose();
        }
    }
}
