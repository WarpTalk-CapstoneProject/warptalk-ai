# Synthesize the evaluation's speech corpus with the TTS voices built into Windows (no download).
# Usage: powershell -File make_speech.ps1 -OutDir <dir>
# Writes <OutDir>/<voice>/NN.wav, 16kHz mono 16-bit, one sentence per file, two speaking rates.
param([Parameter(Mandatory = $true)][string]$OutDir)
Add-Type -AssemblyName System.Speech
$sentences = @(
  "Can everyone hear me now, I think the microphone is working again.",
  "Let's go through the quarterly numbers before we talk about hiring.",
  "The release is scheduled for Thursday, unless the tests fail tonight.",
  "I sent the draft to legal yesterday and they have not replied yet.",
  "We should move the customer demo to next week to be safe.",
  "Honestly the latency looks much better since we changed the region.",
  "Who is taking notes today? I can do it if nobody else wants to.",
  "The second option costs more, but it saves us two months of work.",
  "Please share your screen so we can all look at the same chart.",
  "My connection dropped for a moment, could you repeat the last point?",
  "The translation sounded natural, although the names were spelled wrong.",
  "Let's park that question and come back to it after the break.",
  "I agree with the plan, but we need a fallback if the vendor is late.",
  "Our users in Hanoi reported the same problem on Monday morning.",
  "Remember to update the ticket when the fix is deployed to production.",
  "That is a fair point, and I think the data supports it.",
  "Can we schedule a follow up call with the design team on Friday?",
  "The budget is tight this quarter, so every request needs a clear owner.",
  "I will summarize the decisions and send them out after the meeting.",
  "Thanks everyone, that was productive, see you all next week."
)
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($v in $synth.GetInstalledVoices()) {
  $name = ($v.VoiceInfo.Name -split ' ')[1]
  $dir = Join-Path $OutDir $name
  New-Item -ItemType Directory -Force $dir | Out-Null
  $synth.SelectVoice($v.VoiceInfo.Name)
  $i = 0
  foreach ($rate in @(0, 2)) {
    $synth.Rate = $rate
    foreach ($s in $sentences) {
      $path = Join-Path $dir ("{0:D2}.wav" -f $i)
      $synth.SetOutputToWaveFile($path, $fmt)
      $synth.Speak($s)
      $i++
    }
  }
  $synth.SetOutputToNull()
}
