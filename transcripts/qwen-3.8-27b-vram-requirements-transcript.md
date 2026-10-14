# Transcript: How Much VRAM Does Qwen 3.8 27B Really Need? Every GB, 8GB to 32GB

- **Video URL:** https://www.youtube.com/watch?v=fYLpBnumJZc
- **Creator:** Nibble / Kernel Brief AI
- **Focus:** The 5 VRAM Slices, Q4_K_M variations, Gated Delta linear attention (16 vs 48 layers), KV cache quantization, Speculative Draft Head, and the Memory Bandwidth Law.

---

```text
[00:00:00] Quinn 3.827B
[00:00:03] weighs 55 GB. People are running it on 8
[00:00:06] GB graphics cards. [music] Both of those
[00:00:08] are true, which is the problem.
[00:00:10] Somewhere between 55 and 8. A lot of
[00:00:12] gigabytes go missing and you deserve to
[00:00:14] know where. [music] So, how much VRAM
[00:00:16] does Quinn 3.827B
[00:00:18] really need? We're going to put every
[00:00:20] gigabyte on the board, all five things
[00:00:23] that eat your memory, and find the setup
[00:00:25] that works on your card from 8 GB to 32.
[00:00:28] Here are the receipts. Quinn's own model
[00:00:31] page lists 27.8 billion parameters, each
[00:00:34] stored as a 16- bit number. Napkin math.
[00:00:37] 27.8 billion numbers at 2 bytes each is
[00:00:41] 55.6 GB. That's the full model. If you
[00:00:44] run that at home, I'm impressed and a
[00:00:47] little worried about your power bill.
[00:00:49] It's worth the trouble. On Quinn's own
[00:00:51] table, it scores 61.7 on Bench Pro, a
[00:00:55] coding test, ahead of Claude Opus 4.6 6
[00:00:58] max at 53.4. Those are Quinn's numbers,
[00:01:01] so salt to taste, so people shrink it.
[00:01:03] Unsloth, one of the groups that packs
[00:01:06] models for home use, publishes 25
[00:01:08] versions of this one. I went through all
[00:01:10] of them. The smallest is about 6 GB. The
[00:01:13] biggest is the full 55, and the file is
[00:01:16] only the first of five things that need
[00:01:18] room on your card. First, the file
[00:01:20] itself. Then, the model's notes on your
[00:01:22] conversation and the extras you may not
[00:01:24] know you loaded. then scratch space for
[00:01:27] the math and whatever your screen is
[00:01:28] already using. One bar, five slices.
[00:01:32] Let's fill it in. First, the weights.
[00:01:34] Every one of those 27.8 billion numbers
[00:01:37] is stored at 16 bits. Shrinking a model
[00:01:40] means storing each one with fewer. So,
[00:01:42] how small can it get? Take a guess
[00:01:44] before we do the math. 20 GB, 10, less,
[00:01:48] 6.2. The smallest file squeezes each
[00:01:51] number into about 1.8 bits. a 55 GB
[00:01:55] model in a ninth of the space. The same
[00:01:58] napkin math works for every file.
[00:02:00] Parameters times bits per weight divided
[00:02:03] by 8. So the famous 4-bit versions
[00:02:06] should come out around 14 GB. They
[00:02:08] don't. The popular 4-bit files are 16
[00:02:11] 1/2 to 19 GB, which means a 4-bit file
[00:02:15] is about 5 bits. I'm a 4-bit chip, so I
[00:02:18] took that personally. Here's where the
[00:02:20] extra bit goes. Some parts of a model
[00:02:22] break when you squeeze them. So, the
[00:02:24] people packing these files give those
[00:02:26] parts more bits and the average creeps
[00:02:28] up. And here's a part you might not
[00:02:30] expect. Quen knows 248,000 words and
[00:02:34] pieces of words. And each one comes with
[00:02:36] a list of 5,120
[00:02:39] numbers. It keeps two separate copies of
[00:02:41] that list, one for reading your text and
[00:02:44] one for writing its own. 2 1/2 billion
[00:02:47] parameters. 9% of the model is a
[00:02:49] dictionary. it owns twice. So, a file's
[00:02:52] name tells you roughly how it was
[00:02:54] packed, [music]
[00:02:54] not how big it is. Which brings us to
[00:02:56] the most confusing name in local AI.
[00:02:59] Now, the fine print. Three different
[00:03:02] groups publish a file with Q4KM in its
[00:03:05] name for this model. I checked all
[00:03:07] three. Unsloths is 16 1/2 GB. LM Studios
[00:03:12] is 16.8. And the one from the team
[00:03:14] behind llama.cpp is 19. [music] In bits
[00:03:18] per weight, that's 4.8. 8, 4.9, and 5.6.
[00:03:22] [music]
[00:03:23] All three call themselves 4bit. Same
[00:03:26] name, 2 1/2 GB apart. On a 16 GB card,
[00:03:30] that's the difference between loading
[00:03:32] and not loading. Honest translation,
[00:03:34] Q4KM is a recipe, not a size. That's why
[00:03:38] you'll hear this model called 16 1/2 GB
[00:03:41] at 4 bits and also 19. Both are right.
[00:03:45] So, check the file size, not the name.
[00:03:47] So, what does each bit cost you? Someone
[00:03:49] spent about $3,000 of cloud GPU time to
[00:03:52] find out, so you don't have to. Quizma
[00:03:55] ran the full model and its smaller
[00:03:57] versions through three tests, including
[00:03:59] 89 real tasks in a computer terminal.
[00:04:02] The 4-bit file matched the full model on
[00:04:05] those terminal tasks. [music] Not close,
[00:04:07] matched. But the biggest two-bit file,
[00:04:10] about 11 GB, fell noticeably. It also
[00:04:13] needed about a quarter more tokens to
[00:04:15] finish the same jobs. So, some of its
[00:04:17] speed goes to thinking longer, and the
[00:04:19] one bit files scored about what you'd
[00:04:21] get by guessing. [music] The smallest
[00:04:23] ones scored worse than guessing, which
[00:04:25] takes effort. Here's the catch. Unsloth
[00:04:28] replaced those files on August 19th, so
[00:04:31] the exact versions they tested are gone.
[00:04:33] The pattern still holds. Four bits keeps
[00:04:36] the model. Two bits starts to lose it.
[00:04:39] One bit loses it completely. That's
[00:04:41] slice one. somewhere between 10 and 19
[00:04:44] gigabytes, depending on how much of the
[00:04:46] model you want to keep. I'd keep at
[00:04:47] least four bits if your card allows it.
[00:04:50] Slice two is the one that grows while
[00:04:52] you talk. As the model reads, each
[00:04:54] attention layer keeps a small note on
[00:04:57] every token, so it never has to reread
[00:04:59] your whole conversation. That pile of
[00:05:01] notes is the KV cache. It's why a model
[00:05:04] that fits can still run out of memory
[00:05:06] halfway through a long chat. So, how big
[00:05:09] does the pile get? Take a guess. Quinn
[00:05:12] 3.827B
[00:05:14] 128,000 tokens of conversation. Bigger
[00:05:18] than the model, about the same or
[00:05:20] smaller, about 8 1/2 GB, half the size
[00:05:24] of the 4-bit file. Now, the same
[00:05:26] conversation on last year's Quinn 332B
[00:05:29] about 34 GB, four times as much from a
[00:05:33] model that's barely bigger. So, what
[00:05:36] changed? Here's the trick. Quinn 3.8 8
[00:05:38] has 64 layers, but only 16 of them keep
[00:05:42] notes, every fourth one. The other 48
[00:05:45] are a different kind of layer called
[00:05:46] gated delta. Instead of a note per
[00:05:49] token, each keeps one running summary,
[00:05:52] about 3 megabytes, and rewrites it as it
[00:05:54] reads. 10 tokens or 200,000, the summary
[00:05:58] stays the same size. All 48 together
[00:06:01] come to about 157 megabytes for any
[00:06:05] length of conversation. Oh, that's
[00:06:07] clever. Keep a small summary and get on
[00:06:09] with it. I relate to this on a personal
[00:06:11] level. Napkin math for the 16 that do
[00:06:14] keep notes. [music] For every token,
[00:06:16] each one stores a key and a value. Four
[00:06:18] heads of 256 numbers at two bytes a
[00:06:22] number across 16 layers. That's about 64
[00:06:25] kilobytes a token. Last year's Quinn
[00:06:28] 332B
[00:06:29] stored the same amount per layer, but on
[00:06:32] all 64 layers. Same arithmetic. Four
[00:06:35] times the layers, four times the memory.
[00:06:38] So here's slice 2 by length. 32,000
[00:06:41] tokens, about 2 GB, 128, 8 1/2. The full
[00:06:46] 262,
[00:06:48] 17. You can also shrink the notes
[00:06:50] themselves. Store them at 8 bits instead
[00:06:53] of 16 and the pile halves. At 4 bits,
[00:06:56] it's about a quarter. Now the fine
[00:06:58] print. On the model before this one,
[00:07:00] Quinn 3.6, Six. One tester measured what
[00:07:03] that shrinking costs. At 8 bits, almost
[00:07:06] nothing. At 4 bits, the damage wasn't
[00:07:09] spread out. Nearly all of it landed on
[00:07:11] long documents. So, the setting that
[00:07:14] buys you a long context does its harm on
[00:07:16] long contexts. It's a raincoat that
[00:07:19] leaks only in the rain. My rule, 8-bit
[00:07:22] notes are close to free. 4-bit notes for
[00:07:24] short jobs until someone measures Quinn
[00:07:27] 3.8 itself. Here's the second catch in
[00:07:29] the notes, and it's a default. This
[00:07:31] model thinks before it answers, and out
[00:07:33] of the box, its thinking is set to the
[00:07:35] highest level. Quinn's own page says to
[00:07:38] leave room for up to 262,000
[00:07:41] tokens of thinking. One tester watched
[00:07:44] it think for about 20,000 tokens on a
[00:07:46] single prompt asking for a website. Now,
[00:07:48] open on any card under 24 GB. How much
[00:07:52] room does it give this model by default?
[00:07:55] 4,000 tokens. 4,000 for a model that can
[00:07:59] spend 20,000 of them thinking about one
[00:08:02] web page. That's not a context window.
[00:08:04] That's a sticky note. Olama's own docs
[00:08:07] say coding tools and agents need at
[00:08:09] least 64,000.
[00:08:11] So set it yourself and check it with
[00:08:13] Lama PS. And for simple jobs, turn the
[00:08:16] thinking effort down. The model card
[00:08:18] offers medium and low. That's the two
[00:08:21] big slices, and both are on the model's
[00:08:23] page, but the other three aren't. And
[00:08:26] there why a 16 GB card can say out of
[00:08:28] memory when the math says it fits. Slice
[00:08:31] three, the extras. Quinn 3.8 can read
[00:08:34] images and the part that does it is a
[00:08:37] separate file of about.9 GB. Llama.cpp
[00:08:41] only loads it if you ask. O Lama's
[00:08:43] package comes with it. If you only ever
[00:08:45] send text, that's most of a gigabyte of
[00:08:48] eyes you never use. And if you do want
[00:08:50] images, the llama.cpp CPP team also
[00:08:53] publishes an 8bit version of that file
[00:08:56] about6 GB. The other extra is the draft
[00:08:59] head, a small helper that ships with
[00:09:02] Quinn 3.8 about 1 1/2 GB that can make
[00:09:06] the big model a lot faster. We'll get to
[00:09:08] how. For now, it's a gigabyte and a half
[00:09:11] you might want. [music] Slice four
[00:09:13] scratch space. The program needs room to
[00:09:16] do the math and the guides we read
[00:09:18] budget about a gigabyte for it. Add
[00:09:20] those 48 summaries, another sixth of a
[00:09:23] gigabyte. And slice five, it's the one
[00:09:25] nobody counts. Your screen. If the same
[00:09:28] card runs your desktop and a browser,
[00:09:31] they got there first. Every guide we
[00:09:33] read from people who ran this model
[00:09:35] leaves 1 to 2 GB free. One wrinkle in
[00:09:38] your favor. The 16 GB on your card's box
[00:09:42] are binary gigabytes. In the units file
[00:09:44] sizees use, that's about 17.2.
[00:09:48] For once, the box is being generous. So,
[00:09:51] what fits on a 16 GB card? Let's fill
[00:09:53] the bar. About 17.2 of room. Take off a
[00:09:57] gigabyte for scratch space and summaries
[00:09:59] and a gigabyte for your screen. That
[00:10:01] leaves about 15 for the file and the
[00:10:04] notes. The popular 4bit file is 16 1/2.
[00:10:07] It doesn't fit before you've typed a
[00:10:09] word. The smallest 4bit IQ4XS
[00:10:13] is 14 and 1/4. [music] That leaves 3/4
[00:10:16] of a gigabyte for notes, about 20,000
[00:10:19] tokens at 8 bits, usable, tight. A 3bit
[00:10:23] file, IQ3XXs,
[00:10:26] is about 11 GB. That leaves four for
[00:10:29] notes, over 100,000 tokens at 8 bits.
[00:10:33] That's exactly what one owner of a 16 GB
[00:10:35] card found. The 4-bit files were too
[00:10:38] slow. The three-bit one fit entirely on
[00:10:41] the card, reached 128,000 tokens once
[00:10:44] the notes were squeezed to four bits,
[00:10:46] and wrote 20 to 30 tokens a second.
[00:10:49] Quick question for the comments. Which
[00:10:51] card are you running Quinn 3.827B
[00:10:54] on? [music] 8, 12, 16, 24 GB, or a Mac?
[00:10:59] And which file? Now, speed. An RTX3090
[00:11:03] writes about 42 tokens a second with the
[00:11:06] 4-bit file. an M4 Pro MacBook about 13
[00:11:09] with a three-bit one. Why those numbers?
[00:11:12] Because to write one token, the card has
[00:11:15] to read every weight in the model once.
[00:11:18] Then it reads them all again for the
[00:11:19] next token. So speed has a ceiling. How
[00:11:22] fast your memory can be read divided by
[00:11:25] the size of the file. Napkin math.
[00:11:27] [music] The 3090 reads 936 GB a second.
[00:11:31] And the file is 17 1/2. That's a ceiling
[00:11:34] of 53 tokens a second. [music] It
[00:11:36] measured 41 12, 78% of the ceiling. Real
[00:11:41] runs land somewhere between half and 4
[00:11:43] fths of it. I love this formula. One
[00:11:46] division and most of the speeds in these
[00:11:48] videos stop being mysterious. Which
[00:11:50] means the number to look for on a spec
[00:11:52] sheet for this kind of work isn't the
[00:11:54] compute, it's the memory bandwidth.
[00:11:56] Here's a pair [music] I like. An RTX
[00:11:59] 4060 and an M4 Pro MacBook read memory
[00:12:02] at almost exactly the same speed.
[00:12:05] [music] 272 and 273 GB a second. They've
[00:12:10] never met, but they'd get along. Run the
[00:12:12] formula on any card. A 5090 reads 1,792
[00:12:17] GB a second, a ceiling around 100 with
[00:12:20] the 4-bit file. A 3060 reads 360, about
[00:12:25] 33 with a 3-bit one. The top M4 Max
[00:12:28] reads 546,
[00:12:30] twice the Pro, so twice the ceiling. And
[00:12:33] AMD and Intel cards follow the same
[00:12:35] formula, but their software is
[00:12:37] different, and we didn't find
[00:12:38] measurements we'd trust. So, no numbers
[00:12:40] for me there. And this is why running
[00:12:42] out of VRAMm hurts so much. The part of
[00:12:45] the model that doesn't fit goes to
[00:12:47] system memory, and system memory is
[00:12:49] slow. Two sticks of DDR4 at 3200 read
[00:12:53] about 51 GB a second. The 4060's own
[00:12:56] memory reads 272.
[00:12:59] Every gigabyte you push off the card
[00:13:01] gets read about five times slower. Red
[00:13:04] Stapler ran this model on an 8 gigabyte
[00:13:06] 4060 with more than half of the file in
[00:13:09] system memory. Napkin math says that
[00:13:11] caps a typical DDR4 desktop around five
[00:13:14] tokens a second. He measured four to
[00:13:16] six. At that speed, each of his test web
[00:13:19] pages took between an hour and a half
[00:13:21] and 2 and 1/2 hours. At that point, it
[00:13:24] isn't an assistant. It's a pen pal. And
[00:13:26] it gets worse with a bigger file. One 16
[00:13:29] GB card ran the 5-bit version with most
[00:13:31] of it off the card, 1 and a half tokens
[00:13:33] a second. Someone asked under another
[00:13:35] channel's video whether slower RAM
[00:13:37] matters. It does in direct proportion.
[00:13:40] Memory at 2666 instead of 3200 makes the
[00:13:45] offloaded part about 17% slower. So, how
[00:13:48] do you beat the ceiling? With the draft
[00:13:50] head from slice 3, the small helper
[00:13:52] guesses the next two tokens. Then the
[00:13:54] big model checks both in one pass,
[00:13:57] reading its weights once. When the
[00:13:58] guesses are right, one read gives you
[00:14:01] more than one token. On the same 3090,
[00:14:04] 66 tokens a second on a card whose
[00:14:06] ceiling is 53. Not by driving faster, by
[00:14:10] carrying two packages per trip. Now, the
[00:14:12] fine print. The same tester compared the
[00:14:15] text. With the draft head on, almost
[00:14:17] every long answer came out different
[00:14:19] from the plane version. Not wrong
[00:14:21] necessarily, just different, like asking
[00:14:23] twice. And it costs about a gigabyte and
[00:14:26] a half. On a 24 gigabyte card, I'd turn
[00:14:28] it on. On 16, I'd spend that gigabyte
[00:14:31] and a half on notes. There's one more
[00:14:33] speed nobody puts in the thumbnail. How
[00:14:35] long it takes to read your prompt before
[00:14:37] writing anything. Reading is a different
[00:14:40] job. The card chews through your whole
[00:14:42] prompt in big batches. So, here raw
[00:14:45] compute matters, not just memory. On
[00:14:47] that 24 GB MacBook, a prompt of about
[00:14:50] 4,800 tokens took 48 seconds to read,
[00:14:53] about 100 tokens a second. Paste 20,000
[00:14:56] tokens of code into that Mac, [music]
[00:14:58] and you wait over 3 minutes before the
[00:15:00] first word, long enough to wonder
[00:15:02] whether it's thinking or sulking.
[00:15:04] Someone asked under our strata video
[00:15:06] whether two cards can work together.
[00:15:09] They [music] can. Llama.cpp CPP splits
[00:15:11] the layers between them, but each token
[00:15:14] still reads the whole file, one card
[00:15:16] after the other. So, two cards give you
[00:15:18] room, not speed. So, card by card, 8 GB
[00:15:22] first. The smallest two-bit file is 7
[00:15:25] and 1/4 GB. With notes and scratch
[00:15:28] space, it spills. So, a few layers go to
[00:15:30] system memory. That's the faster option,
[00:15:33] but even the bigger two-bit file slipped
[00:15:35] in the tests. [music] Or do what Red
[00:15:37] Stapler did. Keep the 4-bit file mostly
[00:15:40] off the card at four to six tokens a
[00:15:42] second. My call. It runs and that's
[00:15:45] genuinely impressive. But don't buy an 8
[00:15:47] GB card for this model. If you already
[00:15:49] have one, a smaller model will feel
[00:15:51] better every day. 12 GB, a 3-bit file of
[00:15:55] about 11, notes at 8 bits, [music] and 8
[00:15:58] to 16,000 tokens of context. It's tight
[00:16:00] if the same card runs your screen, but
[00:16:03] the whole model sits on the card with a
[00:16:05] ceiling around 33 tokens a second on a
[00:16:07] 3060 16 GB. The 3bit file with room for
[00:16:12] long notes or the smallest 4bit on a
[00:16:14] short leash. If it were my card, I'd
[00:16:17] take the 3bit and the long notes and 24
[00:16:20] GB. That's where you get the model the
[00:16:22] benchmarks describe. The 4-bit file 17.5
[00:16:25] gigabytes with 64,000 tokens of
[00:16:28] full-size notes or 128,000 at 8 bits.
[00:16:32] About 42 tokens a second on a 3090 and
[00:16:35] 66 with the draft head. 32 GB buys the
[00:16:39] 6-bit file with long notes or the 4-bit
[00:16:41] file with all 262,000 tokens at 8 bits.
[00:16:45] A 5090s ceiling with the 6-bit file is
[00:16:48] around 80 tokens a second. Lovely. If
[00:16:51] you were buying the card anyway, I
[00:16:53] wouldn't buy one just for this. What
[00:16:55] about Macs? A Mac shares one pool of
[00:16:58] memory, and by default, Mac OS lets the
[00:17:01] graphics side use about 3/4 of it. On a
[00:17:03] 24 GB MacBook Pro, that's about 19 GB.
[00:17:07] One tester loaded the 4-bit file anyway.
[00:17:10] It didn't just fail, it took the whole
[00:17:12] Mac down, [music] three kernel panics.
[00:17:15] The 3-bit file ran fine at about 13
[00:17:18] tokens a second. On a 32 GB Mac, the
[00:17:21] 4-bit file should fit with room for
[00:17:23] notes. I wouldn't raise the limit on a
[00:17:25] 24. The Mac has made its feelings clear.
[00:17:28] One more Mac detail. The same tester got
[00:17:30] 32 tokens a second in Apple's own MLX
[00:17:34] software with the draft head on against
[00:17:36] 13 in llama.CP.
[00:17:39] Different software, so we can't say how
[00:17:41] much of that is the draft head. Worth a
[00:17:43] try though. Here's one more way to
[00:17:45] choose. Whatever your card, decide what
[00:17:47] you want most. [music] If you want
[00:17:49] speed, the smallest file that keeps four
[00:17:52] bits, all of it on the card, and the
[00:17:54] draft head on. For long context, three
[00:17:57] bits with the notes at 8 bits. For the
[00:18:00] best answers, four bits or more, and
[00:18:02] short notes, pick two. On anything under
[00:18:05] 24 GB, you don't get all three. So, your
[00:18:09] move. Remember where we started? A 55
[00:18:11] [clears throat] GB model and people
[00:18:13] running it on 8 GB cards. Both true. On
[00:18:17] 8 GB, it runs at two bits or at a few
[00:18:19] tokens a second. 12 fits it at three
[00:18:22] bits with short notes. 16 gives those
[00:18:24] three bits room to think. On 24, it's
[00:18:27] the real thing. Four [music] bits,
[00:18:29] 64,000 tokens of notes, 42 tokens a
[00:18:32] second. And on 32, it's comfortable. And
[00:18:35] the reason it fits it all is that 48 of
[00:18:38] its 64 layers do what I do. Keep a small
[00:18:41] summary and get on with it. My take. For
[00:18:44] this model, the number that matters
[00:18:45] isn't 27 billion. It's 16 layers of
[00:18:49] notes and how fast your card can read
[00:18:51] the rest. [music] I'm Nibble and this is
[00:18:53] Kernel Brief AI. One layer at a time. If
[00:18:56] you want someone reading the fine print
[00:18:58] on your hardware every week, that's
[00:18:59] [music] what this channel is. Subscribe
[00:19:02] and the next model gets the same
[00:19:03] treatment. This model reads all of
[00:19:05] itself for every word. KWN 3.8 flash
[00:19:08] next reads only a sliver, which is how
[00:19:11] Strata runs it on a 12 GB card. That
[00:19:13] video is on your screen right now.
```
