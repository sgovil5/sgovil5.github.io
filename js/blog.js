(function () {
  const params = new URLSearchParams(window.location.search);
  const contentEl = document.getElementById('content');
  const emptyEl = document.getElementById('empty-state');
  const metaEl = document.getElementById('post-meta');

  // Canonical post index (used for both list + metadata)
  const postsIndex = [
    { path: 'posts/Koopman.md', slug: 'Koopman', published: 'December 2025' },
    { path: 'posts/SF.md', slug: 'SF', published: 'November 2025' }
  ];

  function normalizeMdPath(p) {
    if (!p) return '';
    // Keep paths repo-root relative so this works from /blog/<slug>/ pages too.
    return String(p).replace(/^\/+/, '');
  }

  function toRootFetchUrl(mdPath) {
    const norm = normalizeMdPath(mdPath);
    return '/' + norm;
  }

  function fetchMarkdownWithFallback(mdPath) {
    const norm = normalizeMdPath(mdPath);
    const url = '/' + norm;
    return fetch(url, { cache: 'no-cache' }).then(function (res) {
      if (res.ok) return res.text();

      // Backwards compatibility: allow links like posts/Koopman-Blog.md
      if (/-Blog\.md$/i.test(norm)) {
        const altNorm = norm.replace(/-Blog\.md$/i, '.md');
        return fetch('/' + altNorm, { cache: 'no-cache' }).then(function (res2) {
          if (!res2.ok) throw new Error('Failed to load markdown: ' + res.status);
          // Update the caller-visible mdPath for downstream baseDir resolution.
          return res2.text().then(function (txt) {
            return { text: txt, resolvedPath: altNorm };
          });
        });
      }

      throw new Error('Failed to load markdown: ' + res.status);
    }).then(function (result) {
      if (typeof result === 'string') return { text: result, resolvedPath: norm };
      return result;
    });
  }

  function htmlEscape(s) {
    return String(s).replace(/[&<>"']/g, function (ch) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
  }

  function deriveTitle(mdText, fallback) {
    const lines = mdText.split(/\r?\n/);
    for (let k = 0; k < lines.length; k++) {
      const m = lines[k].match(/^#{1,6}\s+(.+)/);
      if (m) {
        let t = m[1].trim();
        t = t.replace(/^Section\s+\d+\s*:\s*/i, '').trim();
        return t;
      }
    }
    return fallback || '';
  }

  function firstContentLines(mdText, count) {
    const lines = mdText.split(/\r?\n/);
    const out = [];
    let i = 0;
    while (i < lines.length && /^#{1,6}\s+/.test(lines[i])) i++;
    for (; i < lines.length && out.length < count; i++) out.push(lines[i]);
    return out.join('\n');
  }

  function protectMath(md) {
    const placeholders = [];
    function makeToken(content) {
      const token = '@@MATH_' + placeholders.length + '@@';
      placeholders.push(content);
      return token;
    }
    function extractDisplayMath(text) {
      let out = '';
      let i = 0;
      while (i < text.length) {
        if (text[i] === '$' && text[i + 1] === '$') {
          let j = i + 2;
          while (j < text.length) {
            if (text[j] === '\\') { j += 2; continue; }
            if (text[j] === '$' && text[j + 1] === '$') { j += 2; break; }
            j++;
          }
          out += makeToken(text.slice(i, j));
          i = j;
        } else {
          out += text[i++];
        }
      }
      return out;
    }
    function extractInlineMath(text) {
      let out = '';
      let i = 0;
      while (i < text.length) {
        if (text[i] === '$' && text[i + 1] !== '$') {
          let j = i + 1;
          while (j < text.length) {
            if (text[j] === '\\') { j += 2; continue; }
            if (text[j] === '$') { j++; break; }
            j++;
          }
          out += makeToken(text.slice(i, j));
          i = j;
        } else {
          out += text[i++];
        }
      }
      return out;
    }
    const protectedMd = extractInlineMath(extractDisplayMath(md));
    return { protectedMd: protectedMd, placeholders: placeholders };
  }

  function configureMarked() {
    if (window.marked && marked.setOptions) {
      marked.setOptions({ breaks: false });
    }
  }

  function typesetMath() {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  }

  function isRelative(url) {
    return !!url &&
      !/^(?:[a-z]+:)?\/\//i.test(url) &&
      url.charAt(0) !== '/' &&
      url.charAt(0) !== '#';
  }

  function setEmptyMessage() {
    emptyEl.innerHTML =
      'Pick a post from the list, or open one directly. Examples: ' +
      '<div><a href="/blog/SF/">Successor Features</a></div>';
  }

  // md can be passed via:
  // - clean URLs: /blog/<slug>/ via window.BLOG_MD_PATH set by the stub page
  // - legacy URLs: /blog.html?md=posts/Foo.md
  let mdPath = (window.BLOG_MD_PATH || params.get('md') || '').trim();

  // If this is the main blog listing page (blog.html) with no md, render cards.
  if (!mdPath) {
    metaEl.textContent = 'Latest posts';
    emptyEl.textContent = 'Loading...';

    Promise.all(postsIndex.map(function (p) {
      return fetchMarkdownWithFallback(p.path).then(function (loaded) {
        const md = loaded.text;
        const baseName = p.path.split('/').pop().replace(/\.[^/.]+$/, '').replace(/[-_]+/g, ' ');
        const fallbackTitle = baseName.replace(/\b\w/g, function (c) { return c.toUpperCase(); });
        const title = deriveTitle(md, fallbackTitle);
        const previewMd = firstContentLines(md, 3);

        const math = protectMath(previewMd);
        configureMarked();
        let summaryHtml = window.marked ? marked.parse(math.protectedMd) : math.protectedMd;
        math.placeholders.forEach(function (original, idx) {
          const token = '@@MATH_' + idx + '@@';
          summaryHtml = summaryHtml.split(token).join(original);
        });

        const linkHref = '/blog/' + encodeURIComponent(p.slug) + '/';
        const dateHtml = (p.published ? ('<div class="post-date">Published: ' + htmlEscape(p.published) + '</div>') : '');
        return ''
          + '<a class="post-card" href="' + linkHref + '">'
          +   '<div class="post-title">' + htmlEscape(title) + '</div>'
          +   dateHtml
          +   '<div class="post-summary">' + summaryHtml + '</div>'
          + '</a>';
      });
    }))
    .then(function (cards) {
      contentEl.innerHTML = cards.join('');
      emptyEl.textContent = '';
      typesetMath();
    })
    .catch(function (err) {
      let hint = '';
      if (window.location && window.location.protocol === 'file:') {
        hint = ' — This page is opened directly from your filesystem. Most browsers block fetch() for local files. Please serve the site locally (e.g., <code>python -m http.server</code> or <code>npx http-server</code>) or view it via GitHub Pages.';
      }
      emptyEl.innerHTML = 'Error: ' + err.message + hint;
    });

    return;
  }

  // Post view
  const mdNorm = normalizeMdPath(mdPath);
  const matched = postsIndex.find(function (p) { return normalizeMdPath(p.path) === mdNorm; }) || null;
  metaEl.textContent = (matched && matched.published) ? ('Published: ' + matched.published) : mdNorm;
  emptyEl.textContent = 'Loading...';

  fetchMarkdownWithFallback(mdNorm)
    .then(function (loaded) {
      const md = loaded.text;
      const resolvedNorm = loaded.resolvedPath || mdNorm;

      const math = protectMath(md);
      configureMarked();
      let html = window.marked ? marked.parse(math.protectedMd) : math.protectedMd;
      math.placeholders.forEach(function (original, idx) {
        const token = '@@MATH_' + idx + '@@';
        html = html.split(token).join(original);
      });

      contentEl.innerHTML = html;

      // Resolve relative resource URLs (images/links) against the markdown file's directory
      try {
        const lastSlash = resolvedNorm.lastIndexOf('/');
        const baseDir = '/' + (lastSlash >= 0 ? resolvedNorm.slice(0, lastSlash + 1) : '');
        Array.prototype.forEach.call(contentEl.querySelectorAll('img'), function (img) {
          const src = img.getAttribute('src');
          if (isRelative(src)) img.setAttribute('src', baseDir + src);
        });
        Array.prototype.forEach.call(contentEl.querySelectorAll('a'), function (a) {
          const href = a.getAttribute('href');
          if (isRelative(href)) a.setAttribute('href', baseDir + href);
        });
      } catch (e) {
        // ignore path rewrite errors
      }

      emptyEl.textContent = '';
      typesetMath();
    })
    .catch(function (err) {
      let hint = '';
      if (window.location && window.location.protocol === 'file:') {
        hint = ' — This page is opened directly from your filesystem. Most browsers block fetch() for local files. Please serve the site locally (e.g., <code>python -m http.server</code> or <code>npx http-server</code>) or view it via GitHub Pages.';
      }
      emptyEl.innerHTML = 'Error: ' + err.message + hint;
      setEmptyMessage();
    });
})();


