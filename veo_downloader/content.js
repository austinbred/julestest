
// content.js
;(function() {
  const INJECT_ID = 'veo-3d-menu-item';

  async function download3DVideo() {
    console.log('📥 Download 3D Game Video clicked');
    const matchId = window.location.pathname.replace(/\/+$/, '').split('/').pop();
    console.log('🔍 matchId =', matchId);
    try {
      const resp = await fetch(`/api/app/matches/${matchId}/videos`, { credentials: 'include' });
      console.log('HTTP', resp.status);
      const data = await resp.json();
      console.log('✅ Full API JSON response:', data);

      const videos = Array.isArray(data) ? data : (data.videos || []);
      const item = videos.find(v => v.render_type === 'panorama' && v.url?.includes('transcode-'));

      if (item?.url) {
        console.log('🔗 Found transcode URL:', item.url);
        const filename = item.url.split('/').pop();
        chrome.runtime.sendMessage({ action: 'downloadVideo', url: item.url, filename });
      } else {
        console.warn('⚠️ No transcode URL found.');
      }
    } catch (err) {
      console.error('❌ Error fetching video list:', err);
    }
  }

  function injectButton(container) {
    console.log('🚧 injectButton() container=', container);
    if (container.querySelector(`#${INJECT_ID}`)) {
      console.log('ℹ️ Download-3D button already injected');
      return;
    }

    const buttons = Array.from(container.querySelectorAll('button.download-item_download-item__TGwnn'));
    console.log('🔍 found download-item buttons:', buttons.length);

    const fullGameBtn = buttons.find(b =>
      b.textContent.trim().toLowerCase().startsWith('download full game')
    );
    console.log('🔍 fullGameBtn =', fullGameBtn);

    let btn;
    if (fullGameBtn) {
      // clone the full-game button so styling matches exactly
      btn = fullGameBtn.cloneNode(true);
      btn.id = INJECT_ID;

      // find the *first* span inside the title-div (that's the bold title)
      const titleDiv = btn.querySelector('div.download-item_download-item-title__4Pq1w');
      const titleSpan = titleDiv ? titleDiv.querySelector('span') : null;
      console.log('🔍 titleDiv =', titleDiv, ' titleSpan =', titleSpan);

      if (titleSpan) {
        titleSpan.textContent = 'Download 3D Game Video';
      } else {
        console.warn('⚠️ could not find titleSpan, skipping text update');
      }

      btn.addEventListener('click', download3DVideo);
      fullGameBtn.parentNode.insertBefore(btn, fullGameBtn.nextSibling);
      console.log('✅ injected new button after fullGameBtn');
    } else {
      console.warn('⚠️ fullGameBtn not found – appending fallback button');
      btn = document.createElement('button');
      btn.id = INJECT_ID;
      btn.className = 'download-item_download-item__TGwnn';
      btn.textContent = 'Download 3D Game Video';
      btn.addEventListener('click', download3DVideo);
      container.appendChild(btn);
      console.log('✅ fallback button appended');
    }
  }

  // watch for the exact .modal_container__mbDPf appearing
  const observer = new MutationObserver(mutations => {
    for (let m of mutations) {
      for (let node of m.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;
        const cont = node.matches('.modal_container__mbDPf')
          ? node
          : node.querySelector('.modal_container__mbDPf');
        if (cont) {
          console.log('🎯 Found modal_container__mbDPf:', cont);
          injectButton(cont);
        }
      }
    }
  });

  console.log('🛠️ Starting observer on document.body …');
  observer.observe(document.body, { childList: true, subtree: true });
})();

