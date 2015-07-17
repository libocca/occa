using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Runtime.InteropServices;

namespace liboccaSharp {

    /// <summary>
    /// utility functions for the OCCA-Setup
    /// </summary>
    public static class Setup {
        
        /// <summary>
        /// This method (should) ensure that the DllImport's 
        /// find the right version of the occa_c.dll resp. libocca.so
        /// </summary>
        /// <param name="alternate_occahome">
        /// if null, the occa directory is assumed to be set by environment variable OCCA_DIR;
        /// otherwise, this string specifies a alterna
        /// </param>
        /// <param name="ReleaseOrDebug">
        /// If true, the dll's from the Release-Configuration are used; otherwise, the Debug-Version is used.
        /// </param>
        public static void SetDllSearchPath(string alternate_occahome = null, bool ReleaseOrDebug = true) {

            OperatingSystem os = Environment.OSVersion;
            PlatformID     pid = os.Platform;

            string occahome;
            if(alternate_occahome != null) {
                occahome = alternate_occahome;
            } else {
                occahome = System.Environment.GetEnvironmentVariable("OCCA_DIR");
            }
            if(occahome == null || occahome.Length <= 0) {
                throw new ArgumentException("Unable to find occa directory. Sepcify either environment variable OCCA_DIR or set the 'alternate_occahome' argument.");
            }
            if(!Directory.Exists(occahome)) {
                throw new ApplicationException("It seems that the occa directory '" + occahome + "' does not exist.");
            }

            switch(pid) {
                case PlatformID.Win32NT:
                case PlatformID.Win32S:
                case PlatformID.Win32Windows:
                case PlatformID.WinCE: {
                        //
                        string _ReleaseOrDebug = ReleaseOrDebug ? "Release" : "Debug";
                        string dllpath;
                        if(IntPtr.Size == 8) {
                            dllpath = Path.Combine(occahome, "visual_studio", "x64", _ReleaseOrDebug);
                        } else if(IntPtr.Size == 4) {
                            dllpath = Path.Combine( occahome, "visual_studio", _ReleaseOrDebug);
                        } else {
                            throw new NotSupportedException();
                        }
                        
                        string dll = Path.Combine(dllpath, "occa_c.dll");
                        if(!File.Exists(dll)) {
                            throw new ApplicationException("Occa DLL does not exist in expected path: '" + dll + "'");
                        }

                        SetDllDirectory(dllpath);

                        break;
                    }
                case PlatformID.MacOSX:
                case PlatformID.Unix: {
                        //
                        string LD_LIBRARY_PATH = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH");
                        string occasopath = Path.Combine(occahome, "lib");
                        if(LD_LIBRARY_PATH != null)
                            LD_LIBRARY_PATH = occasopath + ":" + LD_LIBRARY_PATH;
                        else
                            LD_LIBRARY_PATH = occasopath;
                        Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", LD_LIBRARY_PATH);
                        
                        break;
                    }
                default:
                    throw new NotSupportedException("unknown operating system: " + pid);
            }



        }


        [DllImport("kernel32.dll", CharSet = CharSet.Auto)]
        private static extern void SetDllDirectory(string lpPathName);


    }
}
