window.onload = function() {
  var snowplowTracker = Snowplow.getTrackerUrl('joule.isr.cs.cmu.edu:8081');
  snowplowTracker.enableLinkTracking();
  snowplowTracker.trackPageView();
}
