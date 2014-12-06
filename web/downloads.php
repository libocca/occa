<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA: Downloads') ?>

<?php absInclude("/menu.php"); ?>

<div id="id_body" class="downloads fixed body">

  <h1 class="ui dividing header"> Source Code </h2>

  <p>
    OCCA is open-source and available in our
    <a href="https://github.com/tcew/OCCA2" class="link">Github</a>
    repository.
    <br/>The code is under the MIT license
    (
    <a href="https://github.com/tcew/OCCA2/blob/master/LICENSE" class="link">License</a> and the
    <a href="https://tldrlegal.com/license/mit-license" class="link">TL;DR Legal Explanation</a>
    )
  </p>

  <h1 class="ui dividing header uSpacing4"> Syntax Highlighting </h2>
  <div class="dsm5 indent1">
    <h2 class="ui dividing header"> Emacs </h4>
    <div class="dsm5 indent1">
      There is an
      <a href="https://github.com/tcew/OCCA2/blob/master/editorTools/okl-mode.el" class="link">okl-mode.el</a>
      file in OCCA2/editorTools which sets <?php highlight('okl-mode') ?> for files with the <?php highlight('.okl') ?> extension.
      <br/>The file also contains <?php highlight('occa-mode') ?> for the older version of the kernel language (loads on <?php highlight('.occa') ?> extension files).
    </div>

    <h2 class="ui dividing header"> Vim </h4>
    <div class="dsm5 indent1">
    </div>
  </div>

</div> <!--[ id_body ]-->


<?php addFooter() ?>
