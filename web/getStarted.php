<?php include($_SERVER['DOCUMENT_ROOT'] . '/header.php'); ?>

<?php absInclude("menu.php"); ?>

<div id="id_body" class="getStarted">

  <div id="id_getStartedMenu">

  </div> <!--[ id_getStartedMenu ]-->

  <div id="id_getStartedContents">

    <pre id="editor"><?php echo file_get_contents($_SERVER['DOCUMENT_ROOT'] . "/src/addVectors.okl"); ?></pre>

  </div> <!--[ id_getStartedContents ]-->

</div> <!--[ id_body ]-->

<script src="/library/js/aceMin/ace.js" type="text/javascript" charset="utf-8"></script>
<script>
  var editor = ace.edit("editor");
  editor.setTheme("ace/theme/chrome");
  editor.getSession().setMode("ace/mode/c_cpp");
  editor.setReadOnly(true);
  editor.renderer.setShowGutter(false);
  editor.setHighlightActiveLine(false);
  editor.setDisplayIndentGuides(false);
  editor.setShowPrintMargin(false);
  editor.setOption("maxLines", 20);
</script>

<?php include($_SERVER['DOCUMENT_ROOT'] . '/footer.php'); ?>
