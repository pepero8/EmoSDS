# EmoSDS
EmoSDS: Emotionally Adaptive Spoken dialogue System

※ All the codes are derived from SpeechGPT

## Train results

### After unified fine-tuning(25.1.3) (3 epochs)

- ex1) "This is input: speechgpt/prompt/weather_465_c_0.wav"
	<br>
	```Plain text
	GT:     [sleepy][depressed]  Oh man, that's a bummer. I was really looking forward to soaking up the sun all day.
	EmoSDS: [neutral] The forecast said it might rain later though. Answer: [neutral] Well, let\'s hope it holds off until the evening at least. (temp:0.6)
	EmoSDS: [neutral] The forecast said it might rain later though. Answer: [neutral] Well, let\'s hope it holds off until the evening, fingers crossed. (temp:0.05)
	```

- ex2) "This is input: speechgpt/prompt/weather_465_c_2.wav"
	<br>
	```Plain text
	GT:     [friendly][friendly]  Hopefully it'll hold off until the evening. Let's pack some umbrellas just in case and keep our fingers crossed!
	EmoSDS: [friendly] The forecast said it might rain later though. Answer: [cheerful] Well, let\'s hope it holds off until after our beach time (temp:0.6)
	EmoSDS: [friendly] The forecast said it might rain later though. Answer: [friendly] Let\'s hope it holds off until the evening, fingers crossed (temp:0.05)
	```

### After unified fine-tuning(25.1.8) (3 epochs)

- ex1) "This is input: speechgpt/prompt/weather_465_c_0.wav"
	<br>
	```Plain text
	GT:     [sleepy][depressed]  Oh man, that's a bummer. I was really looking forward to soaking up the sun all day.
	EmoSDS: <neutral> The forecast said it might rain later though. Answer: <friendly> Let\'s hope it holds off until the evening, I\'ll plan a picnic for tomorrow. (temp:0.6)
	EmoSDS: <friendly> The forecast said it might rain later though. Answer: <cheerful> Well, let\'s hope it holds off, fingers crossed (temp:1.0)
	EmoSDS: <neutral> The forecast said it might rain later though. Answer: <friendly> Let\'s hope it holds off until after our picnic. We could use more sunny days. (temp:0.05)
	```

- ex2) "This is input: speechgpt/prompt/weather_465_c_2.wav"
	<br>
	```Plain text
	GT:     [friendly][friendly]  Hopefully it'll hold off until the evening. Let's pack some umbrellas just in case and keep our fingers crossed!
	EmoSDS: <unfriendly> The forecast said it might rain later though. Answer: <friendly> Let\'s hope it holds off until after our picnic. We could use more luck. (temp:0.6)
	EmoSDS: <unfriendly> The forecast says it might rain later though. Answer: <cheerful> Let\'s hope it holds off until after the picnic. (temp:1.0)
	EmoSDS: <friendly> The forecast said it might rain later though. Answer: <friendly> Let\'s hope it holds off until after our picnic. We could use more sunny days. (temp:0.05)
	```

### After unified fine-tuning(25.1.8) (6 epochs)

- ex1) "This is input: speechgpt/prompt/weather_465_c_0.wav"
	<br>
	```Plain text
	GT:     [sleepy][depressed]  Oh man, that's a bummer. I was really looking forward to soaking up the sun all day.
	EmoSDS: <neutral> The forecast did mention it might rain later though. Answer: <friendly> Let\'s hope it holds off until after our picnic (temp:0.6)
	EmoSDS: <friendly> The forecast did mention it might rain later though. Answer: <friendly> Well, let\'s hope it holds off until after our picnic (temp:1.0)
	EmoSDS: <neutral> The forecast did mention it might rain later though. Answer: <neutral> Well, let\'s hope it holds off until after our picnic. (temp:0.05)
	```

- ex2) "This is input: speechgpt/prompt/weather_465_c_2.wav"
	<br>
	```Plain text
	GT:     [friendly][friendly]  Hopefully it'll hold off until the evening. Let's pack some umbrellas just in case and keep our fingers crossed!
	EmoSDS: <neutral> The forecast did mention it might rain later though. Answer: <friendly> Let\'s hope it holds off until after our picnic (temp:0.6)
	EmoSDS: <friendly> The forecast did it might rain later though. Answer: <friendly> Well, that\'ll give us time to pack as we see. I\'ll need to check my schedule first. (temp:1.0)
	EmoSDS: <friendly> The forecast did mention it might rain later though. Answer: <friendly> Let\'s hope it holds off until after our picnic (temp:0.05)
	```