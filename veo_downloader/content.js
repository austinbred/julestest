// content.js
;(function() {
  const INJECT_ID = 'veo-3d-menu-item';
  const DEBUG = true;
  function dlog(...args) { if (DEBUG) console.debug('[VEO3D]', ...args); }

  async function download3DVideo() {
    const matchId = window.location.pathname.replace(/\/+$/, '').split('/').pop();
    dlog('📥 Download 3D Game Video for match', matchId);
    try {
      const resp = await fetch(`/api/app/matches/${matchId}/videos`, { credentials: 'include' });
      dlog('HTTP', resp.status);
      const data = await resp.json();
      dlog('✅ Full API JSON response:', data);

      const stereo = data.find(v => v.render_type === 'panorama' && v.url?.includes('transcode-'));
      const calib  = data.find(v => v.render_type === 'panorama' && v.render_settings);
      if (!stereo || !calib) {
        console.warn('⚠️ Couldn’t find 3D video or calibration blob');
        return;
      }

      const vidUrl   = stereo.url;
      const blobUrl  = calib.render_settings;
      const vidName  = vidUrl.split('/').pop();
      const blobName = blobUrl.split('/').pop();
      dlog('🔗 video:', vidUrl, vidName);
      dlog('🔧 blob :', blobUrl, blobName);

      chrome.runtime.sendMessage({ action: 'downloadFile', url: vidUrl,   filename: vidName  });
      chrome.runtime.sendMessage({ action: 'downloadFile', url: blobUrl,  filename: blobName });
    } catch (err) {
      console.error('❌ Error fetching video list:', err);
    }
  }

  function injectButton(container) {
    dlog('injectButton()', container);
    if (container.querySelector(`#${INJECT_ID}`)) {
      dlog('— already injected');
      return;
    }

    // Find the "Download full game" button
    const fullGameBtn = Array.from(
      container.querySelectorAll('button.download-item_download-item__TGwnn')
    ).find(b => /download full game/i.test(b.textContent));
    if (!fullGameBtn) {
      dlog('— could not find full game button in', container);
      return;
    }

    dlog('— cloning fullGameBtn:', fullGameBtn);
    const btn = fullGameBtn.cloneNode(true);
    btn.id = INJECT_ID;

    // Grab the title span by looking for the first child <span> under the title container
    const titleContainer = btn.querySelector('.download-item_download-item-title__4Pq1w');
    if (titleContainer) {
      const titleSpan = titleContainer.querySelector('span');
      if (titleSpan) titleSpan.textContent = 'Download 3D Game Video';
    }

    btn.addEventListener('click', download3DVideo);
    fullGameBtn.parentNode.insertBefore(btn, fullGameBtn.nextSibling);
    console.log('🚀 [VEO3D] Injected 3D download button');
  }

  // Initial scan in case the modal is already open
  function scanAndInject() {
    const containers = document.querySelectorAll('.modal_container__mbDPf');
    dlog('scanAndInject found', containers.length, 'containers');
    containers.forEach(injectButton);
  }

  scanAndInject();

  // Observe for new modals opening
  new MutationObserver(muts => {
    for (let m of muts) {
      for (let node of m.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;
        dlog('node added:', node);
        if (node.matches('.modal_container__mbDPf')) {
          dlog('— matched container directly');
          injectButton(node);
        } else {
          const inside = node.querySelector?.('.modal_container__mbDPf');
          if (inside) {
            dlog('— found container inside added node');
            injectButton(inside);
          }
        }
      }
    }
  }).observe(document.body, { childList: true, subtree: true });

  dlog('✅ VEO3D content script initialized');
})();

