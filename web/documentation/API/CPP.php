<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA: C++ API') ?>

<?php absInclude("/menu.php"); ?>
<?php absInclude("/documentation/API/menu.php") ?>

<?php absInclude("/sidebarStart.php"); ?>

<div class="entry1"><a href="#Device"          >1. Device          </a></div>
<div class="entry1"><a href="#Memory"          >2. Memory          </a></div>
<div class="entry1"><a href="#Kernel"          >3. Kernel          </a></div>
<div class="entry1"><a href="#Helper Functions">4. Helper Functions</a></div>

<?php absInclude("/sidebarEnd.php"); ?>

<div id="id_body" class="fixed body">

  <h2 id="Device" class="ui dividing header"> Device </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("setup"); ?>
      Content
      <?php nextFunctionAPI("free"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 class="ui dividing header"> Compiler Settings </h4>
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
    <h4 class="ui dividing header"> Memory Functions </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("malloc"); ?>
      Content
      <?php nextFunctionAPI("talloc"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 class="ui dividing header"> Kernel Functions </h4>
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
    <h4 class="ui dividing header"> Streams </h4>
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
    <h4 class="ui dividing header"> Interoperability </h4>
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
    <h4 class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("free"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 class="ui dividing header"> Memory Transfers </h4>
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
    <h4 class="ui dividing header"> Interoperability </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("getMemoryHandle"); ?>
      Content
      <?php nextFunctionAPI("getTextureHandle"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 class="ui dividing header"> Miscellaneous </h4>
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
    <h4 class="ui dividing header"> Constructors / Destructors </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("free"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <div class="dsm5 indent1 dSpacing3">
    <h4 class="ui dividing header"> Miscellaneous </h4>
    <div class="dsm5 indent1">
      <?php startFunctionAPI("mode"); ?>
      Content
      <?php nextFunctionAPI("timeTaken"); ?>
      Content
      <?php endFunctionAPI(); ?>
    </div>
  </div>

  <h2 id="Helper Functions" class="ui dividing header"> Helper Functions </h2>
  <div class="dsm5 indent1 dSpacing3">
    <h4 class="ui dividing header"> Memory </h4>
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
