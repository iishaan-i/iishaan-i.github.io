// Tag filtering with AND logic. Tag color classes are mirrored from the
// rendered post-entry chips so the filter bar matches the cards exactly.
(function () {
  'use strict';

  const activeTags = new Set();

  function buildTagColorMap() {
    const colors = {};
    document.querySelectorAll('.post-entry .post-tags .tag').forEach((chip) => {
      const tag = chip.getAttribute('data-tag');
      if (!tag) return;
      const colorClass = Array.from(chip.classList).find(
        (c) => c.startsWith('tag-') && c !== 'tag'
      );
      if (colorClass) colors[tag] = colorClass;
    });
    return colors;
  }

  function getAllTags() {
    const posts = document.querySelectorAll('.post-entry');
    const tagsSet = new Set();
    posts.forEach((post) => {
      const tagsAttr = post.getAttribute('data-tags');
      if (!tagsAttr) return;
      tagsAttr
        .split(',')
        .map((t) => t.trim())
        .filter((t) => t)
        .forEach((tag) => tagsSet.add(tag));
    });
    return Array.from(tagsSet).sort();
  }

  function createTagFilterBar() {
    const tagFilterBar = document.getElementById('tagFilterBar');
    if (!tagFilterBar) return;

    const colors = buildTagColorMap();
    const allTags = getAllTags();

    allTags.forEach((tag) => {
      const tagButton = document.createElement('span');
      const colorClass = colors[tag];
      tagButton.className = colorClass ? `tag ${colorClass}` : 'tag';
      tagButton.setAttribute('data-tag', tag);
      tagButton.textContent = tag;
      tagButton.addEventListener('click', () => toggleTag(tag));
      tagFilterBar.appendChild(tagButton);
    });
  }

  function toggleTag(tag) {
    if (activeTags.has(tag)) {
      activeTags.delete(tag);
    } else {
      activeTags.add(tag);
    }
    updateTagButtons();
    filterPosts();
  }

  function updateTagButtons() {
    document.querySelectorAll('.tag-filter-bar .tag').forEach((button) => {
      const tag = button.getAttribute('data-tag');
      if (activeTags.has(tag)) {
        button.classList.add('active');
      } else {
        button.classList.remove('active');
      }
    });
  }

  function filterPosts() {
    const posts = document.querySelectorAll('.post-entry');

    if (activeTags.size === 0) {
      posts.forEach((post) => post.classList.remove('hidden'));
      return;
    }

    posts.forEach((post) => {
      const tagsAttr = post.getAttribute('data-tags');
      if (!tagsAttr) {
        post.classList.add('hidden');
        return;
      }
      const postTags = new Set(
        tagsAttr
          .split(',')
          .map((t) => t.trim())
          .filter((t) => t)
      );
      const hasAllTags = Array.from(activeTags).every((tag) => postTags.has(tag));
      if (hasAllTags) {
        post.classList.remove('hidden');
      } else {
        post.classList.add('hidden');
      }
    });
  }

  document.addEventListener('DOMContentLoaded', createTagFilterBar);
})();
