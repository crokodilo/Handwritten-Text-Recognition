import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from PIL import Image
from transformers import RobertaTokenizerFast
import skimage.transform


def caption_image_beam_search(encoder, decoder, image_path, tokenizer, transform, device, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param tokenizer: tokenizer
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()

    k = beam_size
    vocab_size = len(tokenizer)

    # Read image and process
    img = Image.open(image_path)

    image = transform(img).to(device)  # (3, 256, 256)

    with torch.no_grad():
        # Encode
        image = image.unsqueeze(0)  # (1, 3, 256, 256)
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[tokenizer.bos_token_id]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != tokenizer.eos_token_id]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

    return seq, torch.FloatTensor(alphas)


def visualize_att(image_path, seq, alphas, tokenizer, smooth=False):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param tokenizer: tokenizer
    :param smooth: smooth weights?
    """
    
    image = Image.open(image_path)
    a, b = 14*45, 14*7
    image = image.resize([a, b], Image.LANCZOS)

    words = [tokenizer.decode(ind) for ind in seq]

    fig = plt.figure(figsize=(7, 9))

    for t in range(len(words)-1):
        if t > 50:
            break
        ax = fig.add_subplot(int(np.ceil(len(words))), 1, t + 1)
        if t > 0:
            ax.text(0, -15, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=10)
        ax.imshow(image, cmap=cm.Greys_r)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [b, a])
        if t == 0:
            ax.imshow(alpha, alpha=0, cmap=cm.Greys_r)
        else:
            ax.imshow(alpha, alpha=0.3, cmap=cm.Greys_r)
        ax.axis('off')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', default='test-imgs/valid1.png', help='path to image')
    parser.add_argument('--model', '-m', default='BEST_checkpoint.pth.tar', help='path to model')
    parser.add_argument('--tokenizer_directory', '-td', default='tokenizer/',  help='path to tokenizer')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--device', '-d', default=0, type=int, help='device for inference')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_true', help='do not smooth alpha overlay')

    args = parser.parse_args()
    
    device = args.device
    
    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()


    input_size = (80, 900)


    # Read word map
    tokenizer = RobertaTokenizerFast.from_pretrained('tokenizer/')
    
    special_tokens = {tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id}

    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(input_size)

    transform = transforms.Compose([resize, totensor, normalize])

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, tokenizer, transform, args.beam_size)

    decoded_sentence = tokenizer.decode([i for i in seq if i not in special_tokens])
    print(decoded_sentence)

    # Visualize caption and attention of best sequence
    fig = visualize_att(args.img, seq, alphas, tokenizer, args.smooth)
