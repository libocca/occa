$(document).ready(function(){

  $('.ui.api.accordion').accordion({
    collapsible : true,
    active      : false,
    exclusive   : false
  });

  // Remove [active] on dropdown menus
  $('.top.menu .right.menu .topMenu.item .menu a.item')
    .off('click')

  $('.toc.sidebar').sidebar()
    .sidebar({
      overlay: true
    })
    .sidebar('attach events','.ui.launch.button');

  //---[ Highlight ]---
  $('pre.code.block').each(function(i, block) {
    hljs.highlightBlock(block);
  });

});
