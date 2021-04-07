    start = time.time()

tensor_art = torch.LongTensor(padded_articles)
tensor_sum = torch.LongTensor(padded_summaries)

articles_len = len(tensor_art[0])
print(time.time() - start)
exit()
model = Model(dic, articles_len)
model = to_cuda(model)

TEMPORARY CODE
TEMPORARY CODE
TEMPORARY CODE

opt = optim.Adam(params=model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

torch.autograd.set_detect_anomaly(True)

for i in range(100):
    print('Epoch:', i + 1)

    opt.zero_grad()

    out_list, cov_loss = model(tensor_art, tensor_sum)

    # out_list_tmp

    loss = torch.tensor(0.)
    loss = to_cuda(loss)

    # loss_tmp = torch.tensor(0.)
    # loss_tmp = to_cuda(loss_tmp)
    for j in range(out_list.shape[0]):
        # loss += criterion(out_list[j],tensor_sum[j,1:]) # '1:' Remove <SOS>

        k = remove_pad(tensor_sum[j, 1:])
        # out_list_tmp

        loss += criterion(torch.log(out_list[j, :k - 1]), tensor_sum[j, 1:k])
        # out_list[j]
    loss += cov_loss

    # PRINT
    out_string = []
    for word in out_list[j, :k - 1]:
        out_string.append(dic.idx2word[torch.argmax(word)])

    print(out_string)
    # PRINT

    # loss = criterion(out_list,tensor_sum[:,1:])+cov_loss
    print('Loss:', loss)

    loss.backward()
    opt.step()

model(tensor_art[0:3], tensor_sum[0:3])

    #
    # with open(art_name.rstrip(), 'r', encoding="utf8") as art_file:
    #
    #     highlight_flag = False
    #
    #     article = ''
    #     summary = ''
    #
    #     for line in art_file:
    #
    #         if line.startswith('@highlight'):
    #             highlight_flag = True
    #
    #         else:
    #             if line.rstrip():
    #                 if highlight_flag == True:
    #
    #                     # HIGHLIGHT
    #
    #                     highlight = '<SOS> ' + line.rstrip() + ' <EOS> '
    #                     summary = summary + highlight
    #
    #                     highlight_flag = False
    #
    #                 else:
    #
    #                     # ARTICLE
    #
    #                     article = article + line.rstrip() + ' '