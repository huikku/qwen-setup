# Transcript: 128GB VRAM for Local AI: The Real Cost, Speed, & Every Box Tested

- **Video URL:** https://www.youtube.com/watch?v=NA9D49kQFuU
- **Creator:** Nibble / Kernel Brief AI
- **Focus:** 128GB Unified Memory vs. Expert Offloading (Qwen 3.8 Flash-Next), AMD Strix Halo, Mac Studio M5 Max, DGX Spark, and Strata on PC.

---

```text
[00:00:00] Okay, so 6 months ago I finally got fed
[00:00:03] up. I was running a local model at
[00:00:05] something like four tokens per second.
[00:00:07] My laptop fan sounded like it was
[00:00:09] auditioning for a jet engine roll. And
[00:00:11] meanwhile, I was watching people on
[00:00:13] Reddit post screenshots of 60 or 70
[00:00:16] tokens a second on their own machines
[00:00:18] while I sat there thinking, what am I
[00:00:20] even doing wrong? So I did what any
[00:00:22] reasonable person does, which is open a
[00:00:24] browser and type in how much VRAMm do I
[00:00:26] actually need for local AI? And that's
[00:00:29] how I fell into a rabbit hole that I'm
[00:00:31] still trying to climb out of. Everyone
[00:00:33] gives the same answer and it's 128 GB.
[00:00:37] That's the golden threshold where you
[00:00:39] get to run the big models and never pay
[00:00:41] an API bill again. And it sounded a lot
[00:00:43] like buying a dream. And then I started
[00:00:45] pricing it out and I want you to feel
[00:00:47] what I felt here because the cheapest
[00:00:50] legit option I found was $3,000 and the
[00:00:53] most expensive was almost $7,000. And
[00:00:55] somewhere in between those two numbers,
[00:00:56] I started questioning every decision
[00:00:58] I've ever made. But what nobody told me
[00:01:01] is that not all of these options are
[00:01:03] actually buying you the same thing. Some
[00:01:05] of them are really just buying you a
[00:01:06] fourth bit. And that will make a lot
[00:01:08] more sense once we've priced everything.
[00:01:10] So let's go through every route to 128
[00:01:13] gigs. And I mean every route. We'll look
[00:01:16] at the mini PC, the Mac Studio, and the
[00:01:18] Nvidia box. Then four used server cards.
[00:01:21] and then the option that made me feel
[00:01:23] stupid for not trying it first. Now
[00:01:25] before we price anything, we need to
[00:01:27] understand why 128 GB became the number
[00:01:31] in the first place. And it comes down to
[00:01:33] one model which is Quinn 3.8 flash next.
[00:01:37] If you spend any time on the local llama
[00:01:39] subreddit, and I basically live there at
[00:01:41] this point, this model is everywhere and
[00:01:43] it's become the default answer to what
[00:01:46] do I run on a big memory machine. One
[00:01:48] person posted that they haven't called
[00:01:49] the cloud API since the day it came out,
[00:01:51] which gives you a sense of the energy
[00:01:53] around it. And the reason 128 matters is
[00:01:56] boring but real. The 4-bit builds of
[00:01:59] this model land between 94 and 124 GB
[00:02:02] depending on the format, which fits in
[00:02:05] 128 and in nothing smaller. So, the
[00:02:07] logic writes itself where you get 128
[00:02:10] gigs, load the 4-bit file, and you're
[00:02:12] done. But as always, it's not actually
[00:02:14] that simple. Now, this is where it got
[00:02:17] interesting for me and also a little
[00:02:19] embarrassing because I'd been thinking
[00:02:20] about it all wrong. I assumed pretty
[00:02:23] logically. I thought that if a model has
[00:02:25] 125 billion parameters, you need all 125
[00:02:28] billion sitting in fast memory to run it
[00:02:30] fast. And Flash Next just doesn't work
[00:02:32] that way. The model does have 125
[00:02:35] billion parameters. And it also has a 51
[00:02:38] billion parameter lookup table for
[00:02:40] common word sequences, which sounds
[00:02:42] enormous. But for each word it writes,
[00:02:44] it only actually reads about six billion
[00:02:46] of them. And at three bits per weight,
[00:02:48] that's a couple of gigabytes per word.
[00:02:50] Everything else just needs to be
[00:02:52] somewhere your chip can reach when a
[00:02:53] word calls for it. And when we look at
[00:02:55] the model card, it actually says the
[00:02:57] lookup table is more aminable to
[00:02:59] offloading than the rest of the model.
[00:03:01] So there's an engine called Strata that
[00:03:03] keeps the active part on your GPU and
[00:03:05] the rest in normal system RAM. One
[00:03:07] person ran it on an RTX3090,
[00:03:10] which is a card with only 24 GB of VRAM,
[00:03:13] and used 23 gigs of VRAM plus 47 gigs of
[00:03:16] RAM in total. In other words, nothing on
[00:03:19] that machine was 128 gigs. This idea has
[00:03:23] a name, which is expert offloading. The
[00:03:26] way I picture it, the 128 gig box is a
[00:03:28] desk big enough to hold every book you
[00:03:31] own open at once, while the PC setup is
[00:03:33] a smaller desk next to a shelf. And if
[00:03:36] each word only opens a few pages at a
[00:03:38] time, the shelf is fine and you don't
[00:03:40] need the big desk. That was the
[00:03:41] realization that shifted everything for
[00:03:43] me because it means capacity and speed
[00:03:45] are no longer the same purchase. Now,
[00:03:48] let's say we still want the big pool and
[00:03:50] we're buying 128 gigs anyway. There's a
[00:03:53] detail here that will mildly ruin your
[00:03:55] week. On Windows, AMD's Ryzen AI Maxin
[00:03:59] mini PCs, which are one of the main
[00:04:01] routes to 128 gigs, let the GPU side
[00:04:04] claim at most 96 GB by default. So, the
[00:04:08] box was sold as 128 and the other 32
[00:04:11] just sit there with the CPU. On Linux,
[00:04:14] it can go past 100 and some owners
[00:04:16] report getting close to 120. and a 128
[00:04:20] gig Mac does much the same thing because
[00:04:22] Mac OS caps what the GPU can lock at
[00:04:25] around 96 gigs. The developer of one of
[00:04:27] the main Mac engines actually tells you
[00:04:30] to go into the terminal and raise that
[00:04:32] limit by hand before you run anything at
[00:04:34] full context. And when we hold those 96
[00:04:37] gigs against the 4-bit builds, which
[00:04:39] start at 94 gigs before any context
[00:04:42] window at all, it's pretty clear where
[00:04:44] this is going. Even the $6,950
[00:04:47] Nvidia box, the DGX Spark, only sees
[00:04:50] about 122 usable gigs. And since
[00:04:53] Nvidia's own 4-bit file is 124 GB, the
[00:04:57] most popular recipe for running, it
[00:04:59] leaves about 48 gigs on the SSD and
[00:05:01] reads it from there. So, the nearly
[00:05:04] $7,000 machine is also offloading.
[00:05:07] Microsoft at least has the decency to
[00:05:09] mention it because their new laptops say
[00:05:11] in a footnote that the memory the GPU
[00:05:13] can address is less than the total, but
[00:05:16] they never say how much less. And the
[00:05:18] 128 on the box is really a ceiling with
[00:05:21] the default sitting well under it. Now,
[00:05:23] if we've decided we want the pool
[00:05:25] anyway, we should start with the
[00:05:26] cheapest way to get it. That's AMD's
[00:05:28] Ryzen AI Max Plus 395 chip, which
[00:05:32] everyone calls Stricks Halo. It starts
[00:05:35] at $3,99 in a mini PC from a brand
[00:05:38] called Bosgame, while Framework makes
[00:05:40] one for $3,449
[00:05:42] that's currently out of stock. And Minis
[00:05:44] Forum's version is $3,799.
[00:05:48] And it's the same chip with the same 256
[00:05:50] GB per second of memory bandwidth in all
[00:05:53] of them. But the engine you run on this
[00:05:55] machine really matters more than the
[00:05:57] brand of the box. Someone ran Flash Next
[00:05:59] on a Stricks Halo machine with two
[00:06:01] different inference engines 160,000
[00:06:05] tokens deep into a conversation. The
[00:06:07] first engine read that prompt at 113
[00:06:10] tokens a second. And the second one, a
[00:06:12] community fork called GoFo, read it at
[00:06:14] 1,227.
[00:06:16] And that's a 160,000 token prompt
[00:06:20] loading in either 23 minutes or 2
[00:06:22] minutes on the same hardware with the
[00:06:24] same model where the only thing that
[00:06:26] changed was the engine. The newer
[00:06:28] engines go even further. Strata posts 47
[00:06:30] to 54 tokens per second on the 4-bit
[00:06:33] build and around 1,300 tokens per second
[00:06:36] on prompt reading. And Halogen, which is
[00:06:38] another engine a lot of owners swear by,
[00:06:40] post similar numbers. It's closed source
[00:06:43] though, and its forums are mostly people
[00:06:45] asking when that changes, and I'm right
[00:06:47] there with them. A bigger sibling is
[00:06:49] also coming, which is the Ryzen AI Max
[00:06:51] Pro495
[00:06:53] with 192 GB at about $6,799
[00:06:57] from Framework. And that gives you more
[00:06:59] headroom at a similar speed. But for
[00:07:01] right now, if you put Linux on a stricks
[00:07:03] Halo box and pick the right engine, it's
[00:07:05] the cheapest machine that runs the 4-bit
[00:07:07] build well. Now, we get to the two
[00:07:09] expensive options, and they split the
[00:07:11] work in really different ways. The Mac
[00:07:13] Studio with an M5 Max and 128 gigs runs
[00:07:16] $5,99.
[00:07:18] The 40 core GPU config is the only way
[00:07:21] to get 128 gigs and it also brings 614
[00:07:25] GB per second, which is more than twice
[00:07:28] what Stricks Halo has. And going from 64
[00:07:31] to 128 gigs costs $1,600 on the Mac,
[00:07:35] which works out to about $25 a gigabyte.
[00:07:38] on Flash. Next specifically, the
[00:07:40] developer of the MLX serve engine
[00:07:42] reports about 40 tokens per second on
[00:07:44] pros and 75 on code. And the machine is
[00:07:47] quiet and polished with a software story
[00:07:50] on the Mac that's really good. But to be
[00:07:52] fair to the other boxes, nobody has
[00:07:54] published prompt reading speeds for this
[00:07:56] model on this machine yet. And that's a
[00:07:58] real gap because it matters for long
[00:08:00] context agent work. The DGX Spark is the
[00:08:03] CUDA machine at $6,950
[00:08:06] as of early October after Nvidia raised
[00:08:08] the price 74% from launch. It writes at
[00:08:11] about 41 tokens per second median and 27
[00:08:14] on pros and it has a draft ahead trick
[00:08:17] that helps because without it the speed
[00:08:19] drops to 17. Where the Spark used to
[00:08:21] shine was prompt reading. On an older
[00:08:24] model, it was five times faster than
[00:08:26] Stricks Halo, which one benchmark
[00:08:28] translated into saving 83 seconds every
[00:08:30] time a coding agent rereads a 30,000
[00:08:33] token context window. And that was a big
[00:08:35] deal. Now on Flash Next, with today's
[00:08:38] engines, the gap is 1,666
[00:08:42] tokens per second against 1,370.
[00:08:46] So, the 83 second advantage is now about
[00:08:48] 4 seconds, and you're paying a
[00:08:50] significant premium for those 4 seconds.
[00:08:53] And Nvidia just launched a 64 gig Spark
[00:08:55] at $4,999.
[00:08:58] So, two of them for 128 gigs comes to
[00:09:01] $10,000. And I'm not sure that needs any
[00:09:04] commentary from me. On October 7th,
[00:09:06] Nvidia quietly launched something called
[00:09:08] the RTX Spark, which is a completely
[00:09:11] different chip from the DGX Spark,
[00:09:13] despite the name. It's in Windows
[00:09:15] laptops and small desktops. Now,
[00:09:17] Microsoft's Surface Laptop Ultra with
[00:09:20] 128 gigs is $5,899.99.
[00:09:24] And the desktop version, the Surface RTX
[00:09:27] Spark Dev Box, is $5,999.99
[00:09:31] with $128 gigs. Microsoft is claiming
[00:09:35] 2.1 times faster first latency and 6.2
[00:09:39] times faster video than an M5 Pro
[00:09:41] MacBook Pro. But when we look at the
[00:09:43] footnotes, those numbers were measured
[00:09:45] on pre-production 64 GB units. And the
[00:09:49] one pedlop figure uses 4bit math with
[00:09:52] sparsity turned on, which is not how
[00:09:54] you'd normally compare performance.
[00:09:55] Reported memory bandwidth puts it
[00:09:57] somewhere around 273 to 307 GB per
[00:10:01] second, which is next to stricks Halo
[00:10:03] rather than next to the Mac Studio. So,
[00:10:05] my take is to treat it as a laptop that
[00:10:07] holds the 4-bit build and to wait for
[00:10:09] someone outside Microsoft to actually
[00:10:11] time it on this model before spending
[00:10:13] $6,000 on one. Now, we get to the
[00:10:15] chaotic option because the top voted 128
[00:10:18] gig build on Loama this quarter came
[00:10:20] from someone who bought four used Radeon
[00:10:23] Pro V620 cards for $1,400 total. They
[00:10:27] threw in a used EPYC server, CPU, 256
[00:10:31] gigs of DDR4 RAM for $610, and a 1,600
[00:10:36] watt power supply, and walked out with a
[00:10:38] whole machine for around $3,000. On
[00:10:41] Flash Next at 4bit, this thing writes 60
[00:10:44] to 70 tokens per second, which on pros
[00:10:46] is faster than every other box in this
[00:10:49] video. So, the server route wins on row
[00:10:51] output speed, but it also draws 500 to
[00:10:54] 600 W while it's writing. At 4 hours a
[00:10:57] day and 15 cents per kilowatt hour, that
[00:11:00] works out to about $120 a year in
[00:11:02] electricity against around $38 a year
[00:11:05] for a strict Halo box under similar
[00:11:07] load. So, the power bill really is part
[00:11:09] of the price tag. And the deal is gone
[00:11:12] because commenters started reporting
[00:11:13] that V620S climbed to the price of a 32
[00:11:17] gig V100 after that build went viral.
[00:11:20] New alternatives like Intel Arc Pro B70S
[00:11:23] and AMD Radeon AI Pro R9700s
[00:11:27] land around $6,800 for four. Four used
[00:11:31] Instinct Mi50s are $2,300 to $2,600
[00:11:35] before a host machine. And they run on a
[00:11:38] chip AMD put in software maintenance
[00:11:40] mode years ago. and a single RTX Pro
[00:11:43] 6000 Blackwell with 96 gigs runs $8,900
[00:11:47] to $13,250
[00:11:50] if you can find one. So, the server
[00:11:52] route made sense at $3,000, but at
[00:11:54] today's prices, it's a different
[00:11:56] calculation. Now, there's one route that
[00:11:58] none of the 128 GB guides price.
[00:12:01] Strata's own readme lists a test setup
[00:12:03] with an RTX 5070 with 12 GB, a 6 core
[00:12:07] Ryzen 5, and 64 GB of RAM. So, nothing
[00:12:10] fancy at all. On the 2-bit build, it
[00:12:12] writes 94 tokens per second. On the
[00:12:15] 3-bit build, it writes 50 to 62. And it
[00:12:18] reads a 32,000 token prompt at $1,750
[00:12:22] tokens per second. And that's the
[00:12:24] writing speed of the $3,000 to $7,000
[00:12:27] boxes with faster reading than any of
[00:12:29] them on a card you might already own.
[00:12:31] Someone running a 3090, which is a card
[00:12:34] from 2020, in an external enclosure
[00:12:36] connected to a laptop with 64 gigs of
[00:12:39] DDR4, posted 60 to 81 tokens per second
[00:12:43] on the 3-bit build. You might be
[00:12:44] wondering what happens if you just throw
[00:12:46] more RAM at the 4-bit build. One owner
[00:12:48] tried exactly that with an RTX 5090
[00:12:51] laptop GPU paired with 192 GB of RAM.
[00:12:55] The 3.5 bit build ran at 53 to 59 tokens
[00:12:59] per second, but the 4-bit build ran at
[00:13:01] 21 to 33, which is about half the speed.
[00:13:04] So, the PC matches the pool at three
[00:13:07] bits and falls behind at 4. And that's
[00:13:09] the line every route in this video lands
[00:13:12] on one side of. And this is the question
[00:13:14] nobody has put a number on, even though
[00:13:17] it's the one that actually matters. The
[00:13:19] lab that made the 3-bit file says it
[00:13:21] keeps 99.4% 4% of the full model score
[00:13:24] across three benchmark tests. But on one
[00:13:27] of those tests, the full model already
[00:13:29] gets everything right. So 99.4% of
[00:13:32] perfect is still perfect. Owners don't
[00:13:34] fully agree either, where one person
[00:13:36] called the two and threebit builds
[00:13:38] lobomized and another runs the 4-bit
[00:13:40] build across 33090s and says it blows
[00:13:44] 27B out of the water. And nobody has
[00:13:46] published both versions side by side on
[00:13:48] a long agent session, which is the
[00:13:50] actual work you'd buy 128 gigs for. So
[00:13:53] that data gap is real. Now, we can at
[00:13:55] least put a number on the price. If you
[00:13:58] already own a gaming PC with a 12 gig
[00:14:00] card, 64 GB of DDR5 costs about $880 at
[00:14:04] the cheapest listing, and that's your
[00:14:06] 3-bit setup. The cheapest machine that
[00:14:08] runs 4-bit well is the $3,99 Stricks
[00:14:12] Halo, which means the fourth bit cost
[00:14:14] you about $2,200 as an upgrade. If
[00:14:17] you're building from scratch, a 5070,
[00:14:20] the RAM and about $800 for a basic PC
[00:14:22] comes to around $2,400.
[00:14:25] And against the Stricks Halo, the fourth
[00:14:27] bit costs $700. And $700 is the one
[00:14:31] number in this whole video that's easy
[00:14:33] to say yes to if you're building new.
[00:14:35] and 4-bit quality really matters to your
[00:14:37] workflow. Now, I want to be clear about
[00:14:40] the ceiling here because it's lower than
[00:14:42] it looks. Strata runs one model family,
[00:14:45] which is flash next and its variance,
[00:14:47] and the speed gains come directly from
[00:14:49] how that specific model is designed with
[00:14:52] a small active weight set and a lookup
[00:14:54] table built to offload. So, if we point
[00:14:56] the same PC at a different large model,
[00:14:59] we're back to whatever standard lama.cpp
[00:15:01] does with it. Context is tighter, too,
[00:15:03] because one person running a 3.5bit
[00:15:06] build on two 3090s with 128 GB of RAM
[00:15:10] got 130,000 tokens of context and
[00:15:13] mentioned they might go back to Quinn
[00:15:15] 3.827B
[00:15:17] which gave them 256,000 if things don't
[00:15:20] improve and the next model might not
[00:15:22] cooperate at all. Quinn calls Flash Next
[00:15:24] a preview of the Quinn 4 architecture
[00:15:26] and somebody noted the Flash models have
[00:15:28] gotten bigger with every version. So,
[00:15:30] the PC trick relies on design choices
[00:15:32] the next model might not make. But the
[00:15:34] pool doesn't care about any of that
[00:15:36] because a 128 gig box loads whatever the
[00:15:39] next model is and whatever shape it
[00:15:41] takes as long as someone writes an
[00:15:42] engine for it. There are also two kinds
[00:15:45] of buyers the PC route doesn't work for.
[00:15:47] And I want to name them plainly. The
[00:15:49] first is anyone serving multiple people
[00:15:51] or running a swarm of agents at the same
[00:15:53] time where the math changes completely.
[00:15:56] Strata's notes show four R9700 cards
[00:15:59] serving eight clients at 165 tokens per
[00:16:02] second total. And the DGX Spark holds 26
[00:16:05] tokens per second per stream with four
[00:16:07] running, while a gaming PC with one 12
[00:16:10] gig card is a oneperson machine. The
[00:16:12] second is anyone who barely uses it
[00:16:14] because Runpod rents an RTX 6000 with 96
[00:16:18] GB of real VRAM for $29 an hour. And
[00:16:22] your $3,99 Stricks Halo buys you 1,483
[00:16:27] of those hours, which is about a year at
[00:16:29] 4 hours a day. And Alibaba's hosted
[00:16:32] Quinn 3.8 Flash costs 47 cents per
[00:16:35] million output tokens. So $3,000 buys
[00:16:38] you 6.6 6 billion output tokens. At 50
[00:16:41] tokens per second, that's 4 years of
[00:16:44] writing without a break. So, if your box
[00:16:46] is going to sit idle most of the week,
[00:16:48] it makes more sense to rent a bit than
[00:16:50] to buy it. Now, this is how I'd put
[00:16:52] money down today. If you already own a
[00:16:54] gaming PC with a 12 gig card or better,
[00:16:57] buy 64 GB of RAM and run the 3-bit
[00:17:00] build-in strata and try that before you
[00:17:02] spend $3,000 on anything. If you're
[00:17:05] buying one box for 4-bit quality, get
[00:17:07] the Stricks Halo and put Linux on it.
[00:17:09] And pick your engine before you pick
[00:17:11] your brand because the default engine is
[00:17:13] where seven tokens per second came from
[00:17:15] and the right engine is where 50 comes
[00:17:17] from. Buy the Mac Studio if silence and
[00:17:20] writing speed are worth the premium to
[00:17:21] you. And raise its memory limit on day
[00:17:24] one before you do anything else. Buy the
[00:17:26] DGX Spark only if your work specifically
[00:17:29] needs CUDA. Not because you've heard
[00:17:30] good is good, but because you have a
[00:17:32] reason it's required. Wait on the RTX
[00:17:35] Spark until someone outside Microsoft
[00:17:36] has actually timed it on this model. And
[00:17:39] only build four cards if you're serving
[00:17:40] a team and your power bill can absorb
[00:17:42] it. But never judge any of these
[00:17:44] machines on the default engine because
[00:17:46] you'd be paying $3,000 for a number the
[00:17:48] box was never going to hit. I started
[00:17:51] this whole thing frustrated at four
[00:17:53] tokens per second on a laptop. And I
[00:17:55] ended up down a rabbit hole that
[00:17:56] included us server auction sites, Linux
[00:17:59] kernel flags, and at least one moment
[00:18:02] where I seriously considered buying four
[00:18:04] graphics cards and explaining it to my
[00:18:06] partner. What surprised me most wasn't
[00:18:08] the price, but that the bit, that one
[00:18:12] extra bit of precision per weight, is
[00:18:14] actually something you can buy with a
[00:18:16] dollar amount on it. Everything else is
[00:18:18] really just packaging around that
[00:18:20] number. And once you priced a bit, the
[00:18:22] machine you need is often already under
[00:18:24] your desk. If this was useful, subscribe
[00:18:26] because I'm going to follow up with
[00:18:28] hands-on benchmarks once I stop arguing
[00:18:30] with myself about which box to buy. And
[00:18:32] drop a comment if you've got a setup
[00:18:33] running since I really want to know what
[00:18:35] engine you're on. See you in the next
[00:18:37] one.
```
