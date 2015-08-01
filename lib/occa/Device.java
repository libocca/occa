package occa;

import occa.Kernel;
import occa.KernelInfo;
import occa.Stream;
import occa.StreamTag;
import occa.Memory;

public class Device {
  private long handle;

  public static final int usingOKL    = (1 << 0);
  public static final int usingOFL    = (1 << 1);
  public static final int usingNative = (1 << 2);

  public static final KernelInfo noKernelInfo = new KernelInfo();

  public Device(){
    handle = 0;
  }

  public Device(Device d){
    handle = d.handle;
  }

  public Device(String[] infos){
    setup(infos);
  }

  public native void setup(String[] infos);

  public native long memorySize();
  public native long memoryAllocated();

  public native String[] mode();

  public native void setCompiler(String[] compiler_);
  public native void setCompilerEnvScript(String[] compilerEnvScript_);
  public native void setCompilerFlags(String[] compilerFlags_);

  public native String[] getCompiler();
  public native String[] getCompilerEnvScript();
  public native String[] getCompilerFlags();

  public native void flush();
  public native void finish();

  public native void waitFor(StreamTag tag);

  public native Stream createStream();
  public native Stream getStream();
  public native void setStream(Stream s);

  public native StreamTag tagStream();
  public native double timeBetween(StreamTag startTag,
                                   StreamTag endTag);

  public native Kernel buildKernel(String[] str,
                                   String[] functionName,
                                   KernelInfo info);

  public Kernel buildKernel(String[] str,
                            String[] functionName){

    return buildKernel(str, functionName, noKernelInfo);
  }

  public native Kernel buildKernelFromString(String[] content,
                                             String[] functionName,
                                             KernelInfo info,
                                             int language);

  public Kernel buildKernelFromString(String[] content,
                                      String[] functionName){

    return buildKernelFromString(content, functionName, noKernelInfo, usingOKL);
  }

  public Kernel buildKernelFromString(String[] content,
                                      String[] functionName,
                                      KernelInfo info){

    return buildKernelFromString(content, functionName, info, usingOKL);
  }

  public Kernel buildKernelFromString(String[] content,
                                      String[] functionName,
                                      int language){

    return buildKernelFromString(content, functionName, noKernelInfo, language);
  }

  public native Kernel buildKernelFromSource(String[] filename,
                                             String[] functionName,
                                             KernelInfo info);

  public Kernel buildKernelFromSource(String[] filename,
                                      String[] functionName){

    return buildKernelFromSource(filename, functionName, noKernelInfo);
  }

  public native Kernel buildKernelFromBinary(String[] filename,
                                             String[] functionName);

  public native <TM> Memory wrapMemory(TM[] handle_,
                                       long bytes);

  public native <TM> void wrapManagedMemory(TM[] handle_,
                                            long bytes);

  public native <TM> Memory malloc(long bytes,
                                   TM[] src);

  public Memory malloc(long bytes){

    return malloc(bytes, new Object[0]);
  }

  public native <TM> TM[] managedAlloc(long bytes,
                                       TM[] src);

  public <TM> TM[] managedAlloc(long bytes){
    @SuppressWarnings("unchecked")
    TM[] src = (TM[]) new Object[0];

    return managedAlloc(bytes, src);
  }

  public native <TM> Memory mappedAlloc(long bytes,
                                        TM[] src);

  public Memory mappedAlloc(long bytes){

    return mappedAlloc(bytes, new Object[0]);
  }

  public native <TM> TM[] managedMappedAlloc(long bytes,
                                              TM[] src);

  public <TM> TM[] managedMappedAlloc(long bytes){
    @SuppressWarnings("unchecked")
    TM[] src = (TM[]) new Object[0];

    return managedMappedAlloc(bytes, src);
  }

  public native void free();
}