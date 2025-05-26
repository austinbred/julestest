(function() {
  function addDownloadButton() {
    if (document.getElementById('veo-3d-download-btn')) return;
    const btn = document.createElement('button');
    btn.id = 'veo-3d-download-btn';
    btn.textContent = 'Download 3D';
    Object.assign(btn.style, {
      position: 'fixed',
      top: '80px',
      right: '20px',
      zIndex: 1000,
      padding: '8px 12px',
      backgroundColor: '#0064e1',
      color: '#fff',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
    });
    document.body.appendChild(btn);

    btn.addEventListener('click', async () => {
      console.log('📥 Fetching video list from API…');
      const matchId = window.location.pathname.replace(/\/$/, '').split('/').pop();
      try {
        const resp = await fetch(`/api/app/matches/${matchId}/videos`, { credentials: 'include' });
        console.log('HTTP', resp.status);
        const data = await resp.json();
        console.log('✅ Full API JSON response:', data);

        const videos = Array.isArray(data) ? data : (data.videos || []);
        const item = videos.find(
          v => v.render_type === 'panorama' && v.url && v.url.includes('transcode-')
        );

        if (item && item.url) {
          console.log('🔗 Found transcode URL:', item.url);
          const filename = item.url.split('/').pop();
          chrome.runtime.sendMessage({
            action: 'downloadVideo',
            url: item.url,
            filename: filename
          });
        } else {
          console.warn('⚠️ No transcode URL found in response.');
        }
      } catch (err) {
        console.error('❌ Error fetching video list:', err);
      }
    });
  }

  // Initial injection and handle SPA navigation
  addDownloadButton();
  setInterval(addDownloadButton, 1000);
})();

