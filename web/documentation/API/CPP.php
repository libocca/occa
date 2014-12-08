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
      Content
      <?php nextFunctionAPI("free"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-CompilerSettings" class="ui dividing header"> Compiler Settings </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("setCompiler"); ?>
      Content
      <?php nextFunctionAPI("setCompilerEnvScript"); ?>
      Content
      <?php nextFunctionAPI("setCompilerFlags"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-MemoryFunctions" class="ui dividing header"> Memory Functions </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("malloc"); ?>
      Content
      <?php nextFunctionAPI("talloc"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-KernelFunctions" class="ui dividing header"> Kernel Functions </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("buildKernelFromSource"); ?>
      Content
      <?php nextFunctionAPI("buildKernelFromBinary"); ?>
      Content
      <?php nextFunctionAPI("buildKernelFromLoopy"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-Streams" class="ui dividing header"> Streams </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("genStream"); ?>
      Content
      <?php nextFunctionAPI("getStream"); ?>
      Content
      <?php nextFunctionAPI("setStream"); ?>
      Content
      <?php endFunctionAPI(); ?>

      <?php startFunctionAPI("tagStream"); ?>
      Content
      <?php nextFunctionAPI("waitFor"); ?>
      Content
      <?php nextFunctionAPI("timeBetween"); ?>
      Content
      <?php nextFunctionAPI("free(stream)"); ?>
      Content
      <?php endFunctionAPI(); ?>

      <?php startFunctionAPI("flush"); ?>
      Content
      <?php nextFunctionAPI("finish"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Device-Interoperability" class="ui dividing header"> Interoperability </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("cl::wrapDevice"); ?>
      Content
      <?php nextFunctionAPI("cuda::wrapDevice"); ?>
      Content
      <?php nextFunctionAPI("coi::wrapDevice"); ?>
      Content
      <?php endFunctionAPI(); ?>

      <?php startFunctionAPI("wrapStream"); ?>
      Content
      <?php nextFunctionAPI("wrapMemory"); ?>
      Content
      <?php nextFunctionAPI("wrapTexture"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <h2 id="Memory" class="ui dividing header"> Memory </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Constructors" class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("free"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Copies" class="ui dividing header"> Memory Copying </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("copyFrom"); ?>
      Content
      <?php nextFunctionAPI("copyTo"); ?>
      Content
      <?php nextFunctionAPI("asyncCopyFrom"); ?>
      Content
      <?php nextFunctionAPI("asyncCopyTo"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Interoperability" class="ui dividing header"> Interoperability </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("getMemoryHandle"); ?>
      Content
      <?php nextFunctionAPI("getTextureHandle"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Memory-Others" class="ui dividing header"> Others </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("mode"); ?>
      Content
      <?php nextFunctionAPI("bytes"); ?>
      Content
      <?php nextFunctionAPI("swap"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <h2 id="Kernel" class="ui dividing header"> Kernel </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Kernel-Constructors" class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("free"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 id="Kernel-Others" class="ui dividing header"> Others </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("mode"); ?>
      Content
      <?php nextFunctionAPI("timeTaken"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <h2 id="HelperFunctions" class="ui dividing header"> Helper Functions </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 id="HelperFunctions-Memory" class="ui dividing header"> Memory Functions </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("memcpy"); ?>
      Content
      <?php nextFunctionAPI("asyncMemcpy"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

</div> <!--[ id_body ]-->

<?php addFooter() ?>
