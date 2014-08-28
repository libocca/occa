using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace liboccaSharp {
    abstract public class OccaBase : IDisposable {
        internal OccaBase() {
        }

        public IntPtr OccaHandle {
            get;
            protected set;
        }
        
        protected void CheckState() {
            if(this.OccaHandle == IntPtr.Zero)
                throw new NotSupportedException("object disposed");
        }
        
        abstract public void Dispose();
    }
}
