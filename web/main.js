$(document).ready(function(){

  $('.sidebarItem.button')
    .on('touchstart click', function(event) {
      $('.sidebar').sidebar('show');
      event.preventDefault();
    });

  $('.sidebar').sidebar()
    .sidebar({
      overlay: true
    })
    .sidebar('attach events','.ui.sidebarItem.button');

  $('.ui.api.accordion').accordion({
    collapsible : true,
    active      : false,
    exclusive   : false
  });

  // Remove [active] on dropdown menus
  $('.top.menu .right.menu .topMenu.item .menu a.item')
    .off('click')

  //---[ Highlight ]---
  $('pre.code.block').each(function(i, block) {
    hljs.highlightBlock(block);
  });

});
