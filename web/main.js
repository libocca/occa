$(document).ready(function(){

  $('.ui.dropdown')
    .dropdown({
      on      : 'hover',
      duration: 0
    });

  $('pre.code.block').each(function(){
    var editor = ace.edit(this);
    editor.setTheme("ace/theme/chrome");
    editor.getSession().setMode("ace/mode/c_cpp");
    editor.setReadOnly(true);
    editor.renderer.setShowGutter(false);
    editor.setHighlightActiveLine(false);
    editor.setDisplayIndentGuides(false);
    editor.setShowPrintMargin(false);
    editor.setOptions({maxLines: Infinity});
  });

});
