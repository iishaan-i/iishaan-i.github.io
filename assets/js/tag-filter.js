// Tag filtering with AND logic
(function() {
  'use strict';
  
  const activeTags = new Set();
  
  // Collect all unique tags from posts
  function getAllTags() {
    const posts = document.querySelectorAll('.post-entry');
    const tagsSet = new Set();
    
    posts.forEach(post => {
      const tagsAttr = post.getAttribute('data-tags');
      if (tagsAttr) {
        const tags = tagsAttr.split(',').map(t => t.trim()).filter(t => t);
        tags.forEach(tag => tagsSet.add(tag));
      }
    });
    
    return Array.from(tagsSet).sort();
  }
  
  // Create tag filter buttons
  function createTagFilterBar() {
    const tagFilterBar = document.getElementById('tagFilterBar');
    if (!tagFilterBar) return;
    
    const allTags = getAllTags();
    
    allTags.forEach(tag => {
      const tagButton = document.createElement('span');
      tagButton.className = 'tag';
      tagButton.setAttribute('data-tag', tag);
      tagButton.textContent = tag;
      tagButton.addEventListener('click', () => toggleTag(tag));
      tagFilterBar.appendChild(tagButton);
    });
  }
  
  // Toggle tag selection
  function toggleTag(tag) {
    if (activeTags.has(tag)) {
      activeTags.delete(tag);
    } else {
      activeTags.add(tag);
    }
    
    updateTagButtons();
    filterPosts();
  }
  
  // Update visual state of tag buttons
  function updateTagButtons() {
    const tagButtons = document.querySelectorAll('.tag-filter-bar .tag');
    
    tagButtons.forEach(button => {
      const tag = button.getAttribute('data-tag');
      if (activeTags.has(tag)) {
        button.classList.add('active');
      } else {
        button.classList.remove('active');
      }
    });
  }
  
  // Filter posts based on active tags (AND logic)
  function filterPosts() {
    const posts = document.querySelectorAll('.post-entry');
    
    // If no tags are active, show all posts
    if (activeTags.size === 0) {
      posts.forEach(post => {
        post.classList.remove('hidden');
      });
      return;
    }
    
    // AND logic: post must have ALL active tags
    posts.forEach(post => {
      const tagsAttr = post.getAttribute('data-tags');
      if (!tagsAttr) {
        post.classList.add('hidden');
        return;
      }
      
      const postTags = tagsAttr.split(',').map(t => t.trim()).filter(t => t);
      const postTagsSet = new Set(postTags);
      
      // Check if post has ALL active tags
      const hasAllTags = Array.from(activeTags).every(tag => postTagsSet.has(tag));
      
      if (hasAllTags) {
        post.classList.remove('hidden');
      } else {
        post.classList.add('hidden');
      }
    });
  }
  
  // Initialize on page load
  document.addEventListener('DOMContentLoaded', () => {
    createTagFilterBar();
  });
})();

