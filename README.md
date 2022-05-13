# Meta-Optimization of CLIP Guided RGB Tensors

(requires pytorch, torchvision, and clip, clip can be installed with just `pip install git+https://github.com/openai/CLIP`)

## **Abstract**

Using Adam and spherical distance as a loss, images can be iteratively created by optimizing an image tensor based on CLIP embeddings. We try optimizing the CLIP embeddings to guide towards in stages, and *bake* processes for generating from the same prompt in the future, as well as showing how the process can create more structurally cohesive images than guiding only towards the end target embedding. We also see the unexpected side-effect of getting rid of unfavorable patterns created in many CLIP optimization processes.



## **Introduction (skip)**

It feels intuitive to us that during optimization of an image, different structures should be focused on at different times. We try guiding the embedding to something something blah blah i dont know how to write an academic paper if anyone has some resources please refer me because these templates do not help lets just see the results . 

I don't write papers I get the job done and that's that, some guy can write these for me eventually, I have autism I dont understand how this works

We create two processes, named "Baking" and "Microwaving," just because I thought it was funny, for image generation. 

The former being creating a process for a certain prompt that is optimized further than simply optimizing towards the CLIP embedding, and the latter taking a precomputed process and using it again. There's also the process of piecing together parts of other processes to create one for a new unseen prompt but it doesn't work too well, it works! just not too well.

## **Results** (& usage)

Results for microwaving a prompt that has been seen before



to bake a prompt, to microwave a baked prompt, to microwave a new prompt, to normally generate an image

```
python bake.py "a bag of fritos"
python microwave.py baked/a_bag_of_fritos.pt
python auto_microwave.py "a bag of doritos"
python normal.py "a bag of doritos"
```

there is also `batch_bake.py` which just calls bake.py on a list of prompts, which is useful if you just want to let something run all night so you don't feel like the time is wasted (this is a callout to myself)

Baking will log the run to wandb, so if you dont want that functionality comment out lines 9, 11 and 59-63

images were baked at 128x128



| microwaving baked prompt 128x128             | baked prompt optimization                                    | normal clip optimization (control)                           |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| a tulip on a rock                            | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974448719405678642/unknown.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974448735436288030/unknown.png) |
| a rose in a bush                             | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974448890671669318/unknown.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974448935647183008/unknown.png) |
| a young man walking in the middle of a field | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974449012449099846/unknown.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974449025707302912/unknown.png) |



Results for microwaving a prompt that hasn't been seen before, the paths were automatically determined by spherical distance to the original prompt. Reminder that this technique is not what was originally intended, and the focus is mainly on baking and microwaving the same prompts. Also that this is NOT based on the best RGB scripts for higher resolutions, as this was an old personal script I had on hand for generating just pixel art for a discord bot.

(microwaving at half size, full size, then double size)

| microwaving new prompt 64x64  | mean top two paths (30%) + prompt (70%)                      | top 2 spliced randomly (30%) + prompt (70%)                  | normal clip optimization(control)                            | paths used                                |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| an avocado armchair           | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974447124584489030/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974447392885735434/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974446855834452069/output.png) | a lemon, a mushroom                       |
| a dog sitting on the sidewalk | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974442783500795914/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974445154238890034/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974444336672559154/output.png) | the doziest dog, a fluffy animal          |
| a man is eating a burger      | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974445851680333845/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974445516073107517/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974446335786881024/output.png) | a recently discovered disease, a mushroom |

the results of this are definitely subjective but my personal favorite are results from first column. (that's what auto microwave is set to do automatically in the repo, and if you want the other functionality, it's commented out on line 38, uncomment it and comment out line 37)

| microwaving new prompt 128x128 | mean top two paths (30%) + prompt (70%)                      | top 2 spliced randomly (30%) + prompt (70%)                  | normal clip optimization(control)                            |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| an avocado armchair            | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974452674311360532/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974451471682785320/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974453755284193370/output.png) |
| a dog sitting on the sidewalk  | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974460075441807360/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974459463148916777/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974453996448264233/output.png) |
| a man is eating a burger       | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974459729122295849/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974459163952414750/output.png) | ![img](https://cdn.discordapp.com/attachments/910202663327653989/974454668019236945/output.png) |

 The biggest offender in my opinion is just the code that generates the image from the embeddings, it is NOT perfect in the slightest (so feel free to apply this technique to other image generating algorithms)

Here we also see that maybe the randomly spliced paths might perform better? It's not easy to evaluate these even subjectively but to me at least, points for the first: the dog, points for the second: the man and the chair points for the third: none

| microwaving new prompt 32x32  | mean top two paths (30%) + prompt (70%)                      | top 2 spliced randomly (30%) + prompt (70%)                  | normal clip optimization(control)                            |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| an avocado armchair           | ![img](https://cdn.discordapp.com/attachments/908808340023435265/974456949615783936/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974458468461322321/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974455961672642560/output.png) |
| a dog sitting on the sidewalk | ![img](https://cdn.discordapp.com/attachments/908808340023435265/974457437207810058/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974458213720285214/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974456222164070501/output.png) |
| a man is eating a burger      | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974457895771062302/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974458706639089674/output.png) | ![img](https://cdn.discordapp.com/attachments/968373278337032233/974456493514575883/output.png) |

I think here, the only downfall is the amount of prompts that I have baked, (you can see since the closest prompt to "a man eating a burger" is "a recently discovered disease" it makes him.. vomiting... and since the next closest is mushroom it gives him a mushroom cap in the style of a burger)

all possible paths that are baked and included in repo at the time of writing (I probably have added more):

```
an_old_photograph_of_a_dog_made_of_twigs.pt
an_old_photograph_of_a_giraffe_in_a_swimming_pool.pt
a_body_of_water.pt
a_dinosaur_eating_waffles.pt
a_fluffy_animal.pt
a_forest_where_the_trees_are_candy.pt
a_lemon.pt
a_little_monkey.pt
a_lonely_cloud.pt
a_mushroom.pt
a_penguin_in_a_tuxedo.pt
a_pentagonal_green_clock.pt
a_pirate_ship.pt
a_radar_system_examining_aircraft.pt
a_recently_discovered_disease.pt
a_road_sign_with_the_word_'Hello'_on_it.pt
a_rose_in_a_bush.pt
a_steam_train.pt
a_tulip_on_a_rock.pt
a_yellow_schoolbus.pt
a_young_man_walking_in_the_middle_of_a_field.pt
elephants_marching_into_town.pt
indoor_vaporwave_swimming_pool.pt
iridescent_vaporwave_bubbles_#digitalart.pt
more_than_everything,_and_less_than_nothing..pt
the_blue_train_is_approaching_on_the_tracks.pt
the_capybara_is_wearing_a_little_hat.pt
the_doziest_dog.pt
the_end_of_the_world.pt
the_face_on_the_statue_of_liberty.pt
the_liberian_jungle.pt
the_most_powerful_muffin.pt
the_president_of_france.pt
the_sound_of_the_wind.pt
```



### Conclusion	

If using the tool certain ways it can boost visual quality *sometimes*. If you use it for microwaving new prompts, bake at 2-4x resolution and bake as many diverse prompts as possible. This paper also does not explore the effect of having more or less stages (the default here is 10) so you can play with that a bit. My theory though is it works like neural network's parameters, too many with not enough data and it'll overfit (you need a much greater amount of baked prompts for microwaving new prompts to work at all if you have more stages, and the opposite is true as well)



#### Notes

Eventually I will evaluate these results with a pyramid based CLIP image generation technique, as I have one on hand that I wrote not too long ago and don't know why I didn't use it in the first place? It's even shown in a blog post that I wrote, also not too long ago. (https://medium.com/analytics-vidhya/clipdirect-5da430087ea)
