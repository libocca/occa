<?php include($_SERVER['DOCUMENT_ROOT'] . '/main.php'); ?>
<?php addHeader('OCCA: C++ API') ?>

<?php absInclude("/menu.php"); ?>

<div id="id_body">

  <?php absInclude("/documentation/hostAPI/menu.php") ?>

  <pre id="editor"><?php echo file_get_contents($_SERVER['DOCUMENT_ROOT'] . "/src/addVectors.okl"); ?></pre>

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
