using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace liboccaSharp {
    public class Device : IDisposable {

        public Memory malloc(int size) {
            throw new NotImplementedException();
        }

        public void setup(Mode m, int platformID, int deviceID) {
            throw new NotImplementedException();
        }

        public Kernel buildKernelFromSource(string filename, string functionName) {
            //
            throw new NotImplementedException();
        }

        public void Dispose() {
            throw new NotImplementedException();
        }
    }
}
