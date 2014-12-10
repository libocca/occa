<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addSidebarHeader('OCCA: C++ API') ?>

<?php startSidebar(610, 280); ?>

<div class="entry1"><a href="#Device">                  1.  Device                   </a></div>
<div class="entry2"><a href="#Device-Constructors">     1.1 Constructors/Destructors </a></div>
<div class="entry2"><a href="#Device-CompilerSettings"> 1.2 Compiler Settings        </a></div>
<div class="entry2"><a href="#Device-MemoryFunctions">  1.3 Memory Functions         </a></div>
<div class="entry2"><a href="#Device-KernelFunctions">  1.4 Kernel Functions         </a></div>
<div class="entry2"><a href="#Device-Streams">          1.5 Streams                  </a></div>
<div class="entry2"><a href="#Device-Interoperability"> 1.6 Interoperability         </a></div>

<div class="entry1"><a href="#Memory">                  2.  Memory                   </a></div>
<div class="entry2"><a href="#Memory-Constructors">     2.1 Constructors/Destructors </a></div>
<div class="entry2"><a href="#Memory-Copies">           2.2 Memory Copying           </a></div>
<div class="entry2"><a href="#Memory-Interoperability"> 2.3 Interoperability         </a></div>
<div class="entry2"><a href="#Memory-Others">           2.4 Others                   </a></div>

<div class="entry1"><a href="#Kernel">                  3.  Kernel                   </a></div>
<div class="entry2"><a href="#Kernel-Constructors">     3.1 Constructors/Destructors </a></div>
<div class="entry2"><a href="#Kernel-Others">           3.2 Others                   </a></div>

<div class="entry1"><a href="#HelperFunctions">         4.  Helper Functions         </a></div>
<div class="entry2"><a href="#HelperFunctions-Memory">  4.1 Memory Functions         </a></div>

<?php endSidebar(); ?>

<?php absInclude("/menu.php"); ?>
<?php absInclude("/documentation/API/menu.php") ?>

<div id="id_body" class="fixed body">

  <h2 id="Device" class="ui dividing header"> Device </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-Constructors" class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("setup"); ?>
      <div class="dSpacing1 f_rw bold">Functions:</div>

      <pre class="cpp code block">void setup(occa::mode mode,
           const int arg1 = 0, const int arg2 = 0);
void setup(const std::string &mode,
           const int arg1 = 0, const int arg2 = 0);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">occa::mode mode</pre></td><td>
                Enumeration to set the device's <?php highlight('OCCA') ?> mode </br>
                Should be chosen from <code>{OpenMP, OpenCL, CUDA, Pthreads, COI}</code>
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">str::string mode</pre></td><td>
                String setting the device's <?php highlight('OCCA') ?> mode </br>
                Should be chosen from <code>{"OpenMP", "OpenCL", "CUDA", "Pthreads", "COI"}</code> </br>
                The input is <?php highlight('case-insensitive') ?>
              </td></tr>
            <tr class="b t"><td><pre class="cpp api code block">const int arg1</pre></td><td>
                <table class="ui inner table">
                  <thead>
                    <tr><td style="width: 150px;"> OCCA Mode </td><td> Meaning </td></tr>
                  </thead>
                  <tbody>
                    <tr><td> OpenMP   </td><td> Nothing      </td></tr>
                    <tr><td> OpenCL   </td><td> Platform     </td></tr>
                    <tr><td> CUDA     </td><td> Device       </td></tr>
                    <tr><td> Pthreads </td><td> Thread count </td></tr>
                    <tr><td> COI      </td><td> Device       </td></tr>
                  </tbody>
                </table>
              </td></tr>
            <tr class="t"><td><pre class="cpp api code block">const int arg2</pre></td><td>
                <table class="ui inner table">
                  <thead>
                    <tr><td style="width: 150px;"> OCCA Mode </td><td > Meaning </td></tr>
                  </thead>
                  <tbody>
                    <tr><td> OpenMP   </td><td> Nothing                      </td></tr>
                    <tr><td> OpenCL   </td><td> Device                       </td></tr>
                    <tr><td> CUDA     </td><td> Nothing                      </td></tr>
                    <tr><td> Pthreads </td><td> Pinned core for first thread </td></tr>
                    <tr><td> COI      </td><td> Nothing                      </td></tr>
                  </tbody>
                </table>
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Sets the device based on the mode and arguments
      </div>

      <?php nextFunctionAPI("free"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>

      <pre class="cpp code block">void free();</pre>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Frees the device and associated objects </br>
        <?php warning('Warning: Does not free memory and kernels allocated from this device') ?>
      </div>

      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-CompilerSettings" class="ui dividing header"> Compiler Settings </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("setCompiler"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">void setCompiler(const std::string &compiler);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 280px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">const std::string &compiler</pre></td><td>
                Sets the compiler used by the device to build kernels
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Sets the compiler used by the device to build kernels
      </div>

      <?php nextFunctionAPI("setCompilerEnvScript"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">void setCompilerEnvScript(const std::string &compilerEnvScript_);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 360px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">const std::string &compilerEnvScript_</pre></td><td>
                The command in <code>compilerEnvScript_</code> is run prior to compiling kernels
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Set a script to run prior to compiling kernels
      </div>
      <?php nextFunctionAPI("setCompilerFlags"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">void setCompilerFlags(const std::string &compilerFlags);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 320px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">const std::string &compilerFlags</pre></td><td>
                Flags used when compiling a kernel
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Set the flags the device uses when compiling kernels
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-MemoryFunctions" class="ui dividing header"> Memory Functions </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("malloc"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">occa::memory malloc(const uintptr_t bytes,
                    void *source = NULL);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 220px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr class="b"><td><pre class="cpp api code block">const uintptr_t bytes</pre></td><td>
                Allocates continuous memory of size <code>bytes</code> on the device and returns the <code>occa::memory</code> object associated with it.
              </td></tr>
            <tr class="t"><td><pre class="cpp api code block">void *source</pre></td><td>
                If set, the device initializes the allocated memory from <code>source</code>
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Allocates memory on the device
      </div>
      <?php nextFunctionAPI("talloc"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">occa::memory talloc(const int dim,
                    const occa::dim &dims,
                    void *source,
                    occa::formatType type,
                    const int permissions = readWrite);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 240px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr class="b"><td><pre class="cpp api code block">const int dim</pre></td><td>
                <code>dim</code> can be 1 or 2, the dimension of the texture
              </td></tr>
            <tr class="t b"><td><pre class="cpp api code block">const occa::dim &dims</pre></td><td>
                <code>dims.x, dims.y</code> hold the array size in the first and second dimensions respectively</br>
                <code>dims.y</code> is ignored on 1D textures (if <code>dim</code> is 1)
              </td></tr>
            <tr class="t b"><td><pre class="cpp api code block">void *source</pre></td><td>
                Allocating texture memory requires initialization during allocation, where the initial values are loaded from <code>source</code>
              </td></tr>
            <tr class="t b"><td><pre class="cpp api code block">occa::formatType type</pre></td><td>
                Data layout for the texture</br>
                For more information on <code>type</code>, refer to the <a href="" class="link"><code>occa::formatType</code></a> definition
              </td></tr>
            <tr class="t"><td><pre class="cpp api code block">const int permissions</pre></td><td>
                Defaults the permissions for the texture with read-write permissions </br>
                Should be chosen from <code>{occa::readOnly, occa::readWrite}</code>
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Allocates texture memory on the device if available, otherwise it defaults to <code>occa::malloc</code> </br>
        For devices without texture memory, 2D texture objects maintain the dimensions of the texture, allowing you to use the continuous array from the defaulting <code>occa::malloc</code> as a 2D array
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-KernelFunctions" class="ui dividing header"> Kernel Functions </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("buildKernelFromSource"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">occa::kernel buildKernelFromSource(const std::string &filename,
                                   const std::string &functionName,
                                   const kernelInfo &info_ = defaultKernelInfo);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 310px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr class="b"><td><pre class="cpp api code block">const std::string &filename</pre></td><td>
                File containing the kernel source
              </td></tr>
            <tr class="t b"><td><pre class="cpp api code block">const std::string &functionName</pre></td><td>
                Function that will be loaded
              </td></tr>
            <tr class="t"><td><pre class="cpp api code block">const kernelInfo &info_</pre></td><td>
                Compile-time macros and includes can be set through <code>info_</code></br>
                For more information on <code>info_</code>, refer to the <a href="" class="link"><code>occa::kernelInfo</code></a> definition
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Compile kernel from source </br>
        <span class="highlight">Note: Kernel compilation caches the binary in <code>$OCCA_CACHE_DIR</code>, which defaults to <code>~/._occa/</code></span>
      </div>
      <?php nextFunctionAPI("buildKernelFromBinary"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">occa::kernel buildKernelFromBinary(const std::string &filename,
                                   const std::string &functionName);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr class="b"><td><pre class="cpp api code block">const std::string &filename</pre></td><td>
                File containing the kernel source
              </td></tr>
            <tr class="t"><td><pre class="cpp api code block">const std::string &functionName</pre></td><td>
                Function that will be loaded
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Loads the kernel <code>functionName</code> from the binary file <code>filename</code>
      </div>
      <!-- <?php nextFunctionAPI("buildKernelFromLoopy"); ?> -->
      <!-- <div class="dSpacing1 f_rw bold">Function:</div> -->
      <!-- <pre class="cpp code block">;</pre> -->

      <!-- <div class="uSpacing3 f_rw bold"></div> -->
      <!-- <div class="dsm5 indent1"> -->
      <!--   <table class="ui celled api table"> -->
      <!--     <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead> -->
      <!--     <tbody> -->
      <!--       <tr><td><pre class="cpp api code block">ARG1</pre></td><td> -->
      <!--           Description -->
      <!--         </td></tr> -->
      <!--       <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td> -->
      <!--           Description -->
      <!--         </td></tr> -->
      <!--     </tbody> -->
      <!--   </table> -->
      <!-- </div> -->

      <!-- <div class="uSpacing3 f_rw bold">Description:</div> -->
      <!-- <div class="dsm5 indent1"> -->
      <!--   SET DESCRIPTION -->
      <!-- </div> -->
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-Streams" class="ui dividing header"> Streams </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("createStream", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">occa::stream createStream();</pre>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Returns a newly created stream for the device
      </div>
      <?php nextFunctionAPI("getStream", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">occa::stream getStream();</pre>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Returns the current stream the device is using
      </div>
      <?php nextFunctionAPI("setStream", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">void setStream(occa::stream s);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">occa::stream s</pre></td><td>
                Sets the device's current stream to <code>s</code>
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Sets the device's current stream
      </div>
      <?php endFunctionAPI(); ?>

      <?php startFunctionAPI("tagStream", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">tag occa::tagStream()</pre>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Set a tag on the device's current stream and return it</br>
        Can be used with <code>device::waitOn(occa::tag tag_)</code> to sync with the device up until tag <code>tag_</code></br>
        Can also be used to time between tags with <code>device::timeBetween(occa::tag start, occa::tag end)</code>
      </div>
      <?php nextFunctionAPI("waitFor", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">void waitFor(occa::tag tag_);</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">occa::tag tag_</pre></td><td>
                A tag on the device's current stream which <code>waitFor()</code> will wait on
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Waits until the device has processed work until tag <code>tag_</code>
      </div>
      <?php nextFunctionAPI("timeBetween", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">double timeBetween(const occa::tag &startTag,
                   const occa::tag &endTag);</pre>
      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">const occa::tag &startTag</pre></td><td>
                Used for the start time
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">const occa::tag &endTag</pre></td><td>
                Used for the end time
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        Returns the time taken between tags <code>startTag</code> and <code>endTag</code> respectively</br>

      </div>
      <?php nextFunctionAPI("free(stream)", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>

      <?php startFunctionAPI("flush", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("finish", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-Interoperability" class="ui dividing header"> Interoperability </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("cl::wrapDevice", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("cuda::wrapDevice", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("coi::wrapDevice", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>

      <?php startFunctionAPI("wrapStream", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("wrapMemory", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("wrapTexture", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <h2 id="Memory" class="ui dividing header"> Memory </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Constructors" class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("free", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Copies" class="ui dividing header"> Memory Copying </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("copyFrom", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("copyTo", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("asyncCopyFrom", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("asyncCopyTo", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Interoperability" class="ui dividing header"> Interoperability </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("getMemoryHandle", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("getTextureHandle", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Others" class="ui dividing header"> Others </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("mode", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("bytes", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("swap", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <h2 id="Kernel" class="ui dividing header"> Kernel </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Kernel-Constructors" class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("free", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Kernel-Others" class="ui dividing header"> Others </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("mode", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("timeTaken", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <h2 id="HelperFunctions" class="ui dividing header"> Helper Functions </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 id="HelperFunctions-Memory" class="ui dividing header"> Memory Functions </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("memcpy", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php nextFunctionAPI("asyncMemcpy", "notDone"); ?>
      <div class="dSpacing1 f_rw bold">Function:</div>
      <pre class="cpp code block">;</pre>

      <div class="uSpacing3 f_rw bold"></div>
      <div class="dsm5 indent1">
        <table class="ui celled api table">
          <thead><tr><th style="width: 200px;">Argument</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
            <tr class="b"><td><pre class="cpp api code block">ARG1</pre></td><td>
                Description
              </td></tr>
          </tbody>
        </table>
      </div>

      <div class="uSpacing3 f_rw bold">Description:</div>
      <div class="dsm5 indent1">
        SET DESCRIPTION
      </div>
      <?php endFunctionAPI(); ?>
    </div>
  </div>

</div> <!--[ id_body ]-->

<?php addFooter() ?>
