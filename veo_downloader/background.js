chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'downloadVideo' && message.url) {
    chrome.downloads.download({
      url: message.url,
      filename: message.filename,
      saveAs: false
    }, downloadId => {
      if (chrome.runtime.lastError) {
        console.error('❌ Download failed:', chrome.runtime.lastError);
      } else {
        console.log('✅ Download initiated, id:', downloadId);
      }
    });
  }
});
