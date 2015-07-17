using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace liboccaSharp {
    public class occaDim : IEquatable<occaDim> {
        internal occaDim() { 
        }

        UIntPtr _x;
        UIntPtr _y;
        UIntPtr _z;

        public UIntPtr x {
            get {
                return _x;
            }
            set {
                _x = value;    
            }
        }

        public UIntPtr y {
            get {
                return _y;
            }
            set {
                _y = value;
            }
        }
        public UIntPtr z {
            get {
                return _z;
            }
            set {
                _z = value;
            }
        }

        public int xi {
            get {
                return (int)_x;
            }
            set {
                _x = (UIntPtr)value;
            }
        }

        public int yi {
            get {
                return (int)_y;
            }
            set {
                _y = (UIntPtr)value;
            }
        }
        public int zi {
            get {
                return (int)_z;
            }
            set {
                _z = (UIntPtr)value;
            }
        }
        

        public int this[int i] {
            get {
                switch(i) {
                    case 0: return (int)_x;
                    case 1: return (int)_y;
                    case 2: return (int)_z;
                    default: throw new IndexOutOfRangeException();
                }
            }
            set {
                switch(i) {
                    case 0: _x = (UIntPtr)value; return;
                    case 1: _y = (UIntPtr)value; return;
                    case 2: _z = (UIntPtr)value; return;
                    default: throw new IndexOutOfRangeException();
                }
            }
        }

        /*
        public UIntPtr this[int i] {
            get {
                switch(i) {
                    case 0: return _x;
                    case 1: return _y;
                    case 2: return _z;
                    default: throw new IndexOutOfRangeException();
                }
            }
            set {
                switch(i) {
                    case 0: _x = value; return;
                    case 1: _y = value; return;
                    case 2: _z = value; return;
                    default: throw new IndexOutOfRangeException();
                }
            }
        }
        */
        public bool Equals(occaDim other) {
            return (other._x == this._x && other._y == this._y && other._z == this._z);
        }
    }
}
