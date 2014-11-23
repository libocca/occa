$(document)
  .ready(function() {
    $('.ui.dropdown')
      .dropdown({
        on      : 'hover',
        delay   : {show: 0},
        duration: 150
      })
    ;
  });
