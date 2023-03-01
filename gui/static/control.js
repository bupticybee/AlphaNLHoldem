var socket;
$(document).ready(function() {
    // The http vs. https is important. Use http for localhost!
    socket = io.connect('http://' + document.domain + ':' + location.port);

    // Button was clicked
    document.getElementById("send_button").onclick = function() {
        // Get the text value
        var text = document.getElementById("textfield_input").value;

        // Update the chat window
        document.getElementById("chat").innerHTML += "You: " + text + "\n\n";

        // Emit a message to the 'send_message' socket
        socket.emit('send_message', {'text':text});

        // Set the textfield input to empty
        document.getElementById("textfield_input").value = "";
    }

    // Message recieved from server
    socket.on('message_from_server', function(data) {
        var text = data['text'];
        console.log(data);
        set_board(data["data"])
        //document.getElementById("chat").innerHTML += "Server push a new state." + text + "\n";
    });

    socket.on('debug_info', function(data) {
        var text = data['text'];
        var input = document.getElementById("chat");
        input.focus(); // that is because the suggest can be selected with mouse
        input.innerHTML += text + "\n";
        input.scrollTop = input.scrollHeight;

    });

    var config = {
        type: Phaser.AUTO,
        parent: 'game_container',
        width: 1024,
        height: 720,
        backgroundColor: '0xbababa',
        scene: {
            preload: preload,
            update: update_stuff
        }
    };

    var game = new Phaser.Game(config);
    var scene;
    var total_reward = 0;
    var total_match = 0;

    function send_action_to_server(action_id){
        socket.emit('send_message', {'text':"action_from_client","action_id":action_id});
    }

    function onAction(action_ind){
        socket.emit('send_message', {'text':"action_from_client","action_id":action_ind});
    }

    function set_board (infos)
    {
        if (scene.reg_images){
            for(var one_image of scene.reg_images){
                one_image.destroy();
            }
        }
        scene.reg_images = [];

        var done = infos["done"]
        console.log("done" + done)
        var ai_id = infos["ai_id"]
        var human_id = 1 - ai_id
        var human_hand = human_id == 0 ? infos["hand_p0"]:infos["hand_p1"]
        var ai_hand = ai_id == 0 ? infos["hand_p0"]:infos["hand_p1"]

        if (done === true){
            total_reward += infos["payoffs"][human_id]
            total_match += 1
        }

        var x = 150;
        var y = 170;
        // opponent_hands
        for (var i = 0; i < 2; i++)
        {
            let one_hand = ai_hand[i];
            let img_added = null;
            if (done === false){
                img_added = scene.add.image(x, y,"card_back").setInteractive();
            }else{
                img_added = scene.add.image(x, y,one_hand).setInteractive();
            }
            img_added.setScale(2);
            img_added.custom_info = {"what":"custom info here"};

            img_added.on('pointerover',function(pointer){
                img_added.setScale(3);
            })
            img_added.on('pointerout',function(pointer){
                img_added.setScale(2);
            })
            scene.reg_images.push(img_added);
            x += 100;
        }


        var x = 150;
        var y = 550;

        // hero_hands
        for (var i = 0; i < human_hand.length; i++)
        {
            const one_hand = human_hand[i];


            let img_added = scene.add.image(x, y,one_hand).setInteractive();
            img_added.setScale(2);

            img_added.on('pointerover',function(pointer){
                img_added.setScale(3);
                img_added.depth_bakup = img_added.depth;
                img_added.depth = 100;
            })
            img_added.on('pointerout',function(pointer){
                img_added.setScale(2);
                img_added.depth = img_added.depth_bakup;
            })

            scene.reg_images.push(img_added);
            // image scale would be 86 * 117
            x += 100;
        }

        var x = 150;
        var y = 350;
        var public = infos["public"]

        // public cards
        for (var i = 0; i < public.length; i++)
        {
            const one_public = public[i];


            let img_added = scene.add.image(x, y,one_public).setInteractive();
            img_added.setScale(2);

            img_added.on('pointerover',function(pointer){
                img_added.setScale(3);
                img_added.depth_bakup = img_added.depth;
                img_added.depth = 100;
            })
            img_added.on('pointerout',function(pointer){
                img_added.setScale(2);
                img_added.depth = img_added.depth_bakup;
            })

            scene.reg_images.push(img_added);
            // image scale would be 86 * 117
            x += 100;
        }

        scene.reg_images.push(scene.add.text(170,660 , 'My stake : ' + (human_id == 0?infos["stakes"][0]:infos["stakes"][1]), { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));
        scene.reg_images.push(scene.add.text(170,675 , 'My Chip in pot: ' + (human_id == 0?infos["chip"][0]:infos["chip"][1]), { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));
        scene.reg_images.push(scene.add.text(170,40 , 'AI Stake: ' + (ai_id == 0?infos["stakes"][0]:infos["stakes"][1]), { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));
        scene.reg_images.push(scene.add.text(170,55 , 'AI Chip in pot: ' + (ai_id == 0?infos["chip"][0]:infos["chip"][1]), { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));
        scene.reg_images.push(scene.add.text(170,250 , 'Pot : ' + infos["pot"], { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));
        scene.reg_images.push(scene.add.text(30,30 , 'Total Win : ' + total_reward, { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));
        scene.reg_images.push(scene.add.text(30,45 , 'Total Match : ' + total_match, { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));
        scene.reg_images.push(scene.add.text(30,60 , 'Win/Match: ' + parseInt(total_reward / total_match), { fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif' }));

        var x = 600;
        var y = 30;
        {
            for (var i = 0; i < infos["action_recoards"].length;i ++) {
                scene.reg_images.push(scene.add.text(x, y + i * 15, (ai_id == infos["action_recoards"][i][0]?'AI ':"Human ") + infos["action_recoards"][i][1],
                    {fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif'}));
            }
            if(done === true){
                human_payoff = infos["payoffs"][human_id]
                scene.reg_images.push(scene.add.text(x, y + i * 15,"Human payoff: " + human_payoff,
                    {fontFamily: 'Georgia, "bold 32px Goudy Bookletter 1911", Times, serif'}));
            }
        }


        var x = 380;
        var y = 490;
        // Buttons
        {
            var legal_actions = infos["legal_actions"]
            for (var i = 0; i < legal_actions.length; i++) {
                if ( legal_actions[i][1]) {
                    let ix = parseInt(i)
                    let whatButton = scene.add.text(x, y + i * 40, legal_actions[i][0])
                        .setOrigin(0.5)
                        .setPadding(10)
                        .setStyle({backgroundColor: '#111'})
                        .setInteractive({useHandCursor: true})
                        .on('pointerdown', function(pointer){
                            onAction(ix);
                        })
                        .on('pointerover', () => whatButton.setStyle({fill: '#f39c12'}))
                        .on('pointerout', () => whatButton.setStyle({fill: '#FFF'}))
                    scene.reg_images.push(whatButton);
                }else{
                    let whatButton = scene.add.text(x, y + i * 40, legal_actions[i][0])
                        .setOrigin(0.5)
                        .setPadding(10)
                        .setStyle({backgroundColor: '#808080'})
                    scene.reg_images.push(whatButton);
                }
            }
        }
    }

    var names = [
    ]
    for(let i of "23456789TJQKA"){
        for(let j of "CDHS"){
            names.push(j + i)
        }
    }

    var game_object_names = [
    ]

    function preload ()
    {
        scene = this;
        for (var i = 0; i < names.length; i++) {
            var one_name = names[i];
            this.load.image(one_name, "static/cards/" + one_name + ".png");
        }
        this.load.image("card_back", "static/cards/card_back.png");
    }

    function update_stuff (infos)
    {
    }
    socket.emit('send_message', {'text':"start"});

});
