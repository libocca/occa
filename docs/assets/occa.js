var occa = occa || {};

occa.languageLabels = {
  cpp: 'C++',
  okl: 'OKL',
};

occa.getLanguageLabel = (language) => (
  occa.languageLabels[language] || language.toUpperCase()
);

//---[ Header & Footer ]----------------
occa.addFooter = (content) => (
  content
    + '\n'
    + '---\n'
    + '<span id="copyright">\n'
    + `  © Copyright 2014 - ${(new Date()).getFullYear()}, David Medina and Tim Warburton.\n`
    + '</span>\n'
);
//======================================

//---[ Include ]------------------------
occa.addIncludes = (vm, content) => {
  return content;
}
//======================================

//---[ Tabs ]---------------------------
occa.markdown = {
  space: () => (
    ''
  ),
  text: ({ text }) => (
    `<p>${text}</p>`
  ),
  list_start: () => (
    '<ul>'
  ),
  list_end: () => (
    '</ul>'
  ),
  list_item_start: () => (
    '<li>'
  ),
  list_item_end: () => (
    '</li>'
  ),
};

occa.markdown.code = ({ lang, text }) => {
  // Remove indentation
  const initIndent = text.match(/^\s*/)[0];
  if (initIndent.length) {
    const lines = text .split(/\r?\n/);
    const isIndented = lines.every((line) => (
      !line.length
      || line.startsWith(initIndent)
    ));

    if (isIndented) {
      text = lines.map((line) => (
        line.substring(initIndent.length)
      )).join('\n');
    }
  }

  // Generate highlighted HTML
  const styledCode = Prism.highlight(text,
                                     Prism.languages[lang],
                                     lang);

  // Wrap around pre + code
  return (
    (
      `<pre data-lang="${occa.getLanguageLabel(lang)}">`
        + `<code class="lang-${lang}">`
        + `${styledCode}\n`
        + '</code>'
        + '</pre>'
    )
      .replace(/[*]/g, '&#42;')
      .replace(/[_]/g, '&#95;')
  );
}

occa.tokenToMarkdown = (token) => {
  const { type } = token;
  if (type in occa.markdown) {
    return occa.markdown[token.type](token);
  }
  console.error(`Missing token format for: ${token.type}`);
  return '';
};

occa.mergeTextTokens = (tokens) => {
  const newTokens = [];
  let texts = [];
  for (var i = 0; i < tokens.length; ++i) {
    const token = tokens[i];
    if (token.type === 'text') {
      texts.push(token.text);
      continue;
    }
    if (texts.length) {
      newTokens.push({
        type: 'text',
        text: texts.join(' '),
      });
      texts = [];
    }
    newTokens.push(token);
  }
  // Join the tail texts
  if (texts.length) {
    newTokens.push({
      type: 'text',
      text: texts.join(' '),
    });
  }
  return newTokens;
};

occa.tokensToMarkdown = (tokens) => {
  tokens = occa.mergeTextTokens(tokens);
  return (
    tokens
      .map(occa.tokenToMarkdown)
      .join('\n')
  );
};

occa.getTab = ({ tab, content }) => (
  `      <md-tab id="${tab}" md-label="${tab}">\n`
    + occa.tokensToMarkdown(content)
    + '      </md-tab>'
);

occa.getTabs = (namespace, tabs) => {
  const content = tabs.map(occa.getTab).join('\n');

  const tab     = `vm.$data.tabNamespaces['${namespace}']`;
  const onClick = `(tab) => vm.onTabChange('${namespace}', tab)`;

  return (
    '<template>\n'
      + '  <div>\n'
      + '    <md-tabs\n'
      + '      md-dynamic-height="true"\n'
      + `      v-bind:md-active-tab="${tab}"\n`
      + `      @md-changed="${onClick}"\n`
      + '    >\n'
      + `${content}\n`
      + '    </md-tabs>\n'
      + '  </div>\n'
      + '</template>\n'
  );
};

occa.parseTabs = (namespace, content) => {
  const parts = marked.lexer(content);
  const newParts = [];

  // Skip begin/end of list
  for (var i = 1; i < (parts.length - 1); ++i) {
    var stackSize = 1;

    // Skip loose_item_start;
    ++i;
    const tab = parts[i++].text;
    const start = i++;

    while ((i < (parts.length - 1)) && (stackSize > 0)) {
      switch (parts[i].type) {
      case 'list_item_start':
        ++stackSize;
        break;
      case 'list_item_end':
        --stackSize;
        break;
      }
      ++i;
    }

    // Don't take the token after list_item_end
    --i;

    newParts.push({
      tab,
      content: parts.slice(start, i),
    });
  }

  if (!newParts.length) {
    return [];
  }

  if (!(namespace in vm.$data.tabNamespaces)) {
    Vue.set(vm.tabNamespaces, namespace, newParts[0].tab);
  }

  return occa.getTabs(namespace, newParts);
};

occa.addTabs = (content) => {
  const re = /\n::: tabs (.*)\n([\s\S]*?)\n:::(\n|$)/g;
  const parts = [];
  var lastIndex = 0;
  while ((match = re.exec(content)) != null) {
    const [fullMatch, namespace, tabContent] = match;

    parts.push(content.substring(lastIndex, match.index));
    parts.push(occa.parseTabs(namespace, tabContent));

    lastIndex = match.index + fullMatch.length;
  }
  parts.push(content.substring(lastIndex));

  return parts.join('\n');
};
//======================================

occa.mainPages = new Set([
  'README.md',
  'history.md',
  'team.md',
  'gallery.md',
]);

occa.docsifyPlugin = (hook, vm) => {
  hook.init(() => {
    Prism.languages.okl = Prism.languages.extend('cpp', {
      annotation: {
        pattern: /@[a-zA-Z][a-zA-Z0-9_]*/,
        greedy: true,
      },
    });
  });

  hook.beforeEach((content) => {
    content = occa.addIncludes(vm, content);
    content = occa.addTabs(content);
    content = occa.addFooter(content);
    return content;
  });

  hook.doneEach(() => {
    const dom = document.querySelector('body');
    const file = vm.route.file;
    // Add API styling
    if (!file.startsWith('api/')) {
      dom.classList.remove('api-container');
    } else {
      dom.classList.add('api-container');
    }
    // Close sidebar
    if (occa.mainPages.has(file)) {
      dom.classList.add('no-sidebar');
    } else {
      dom.classList.remove('no-sidebar');
    }
  });
};
