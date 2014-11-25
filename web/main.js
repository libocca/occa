$(document).ready(function(){


  $('.ui.dropdown')
    .dropdown({
      on      : 'hover',
      duration: 0
    });

  // Remove [active] on dropdown menus
  $('.top.menu .right.menu .topMenu.item .menu a.item')
    .off('click')

  $('.toc.sidebar').sidebar()
    .sidebar({
      overlay: true
    })
    .sidebar('attach events','.ui.launch.button');

  $('pre.code.block').each(function(){
    var editor = ace.edit(this);
    editor.setTheme('ace/theme/chrome');

    var language = $(this).attr('language');

    if(language[0] != '/')
      editor.getSession().setMode('ace/mode/' + language);
    else
      editor.getSession().setMode(language);

    editor.setReadOnly(true);
    editor.renderer.setShowGutter(false);
    editor.setHighlightActiveLine(false);
    editor.setDisplayIndentGuides(false);
    editor.setShowPrintMargin(false);
    editor.setOptions({maxLines: Infinity});
  });

});
