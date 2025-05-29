// background.js
chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.action === 'downloadFile' && msg.url && msg.filename) {
    console.log('⬇️ Starting download:', msg.filename, msg.url);

    chrome.downloads.download({
      url:            msg.url,
      filename:       msg.filename,
      conflictAction: 'uniquify'
    }, downloadId => {
      if (chrome.runtime.lastError) {
        console.error('❌ Download failed:', chrome.runtime.lastError.message);
      } else {
        console.log('✅ Download started, id:', downloadId);
      }
    });
  }
});
