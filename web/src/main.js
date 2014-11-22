$(document)
  .ready(function() {

    $('.top.wrapper .menu .item').tab({
      context: '#id_body'
    });

    $('.documentation.tab .menu .documentation.menu .item').tab({
      context: '#id_documentation'
    });

  });
