import os, time
import torch

def load_checkpoint(model, optimizer, training_logs, checkpoint_path=None, device='cpu'):
    epoch_number = 0
    best_vloss = float('inf')
    if checkpoint_path:
        if os.path.exists(checkpoint_path + 'model.pth'):
            model.load_state_dict(torch.load(checkpoint_path + 'model.pth', weights_only=True, map_location=device))
            print(model.fc3)

        if os.path.exists(checkpoint_path + 'opt.pth'):
            optimizer.load_state_dict(torch.load(checkpoint_path + 'opt.pth', weights_only=True, map_location=device))

        if os.path.exists(checkpoint_path + 'training_logs.pth'):
            training_logs = torch.load(checkpoint_path + 'training_logs.pth', weights_only=True)
            epoch_number = len(training_logs['train_loss'])
            best_vloss = min(training_logs['validate_loss'])
    
    for i in range(epoch_number):
        print(f"Epochs {i+1}".ljust(10), end='')
        for key in training_logs.keys():
            print(f"{key}: {training_logs[key][i]:.5f}", end=" ")
        print()
        print("-"*80)

    return training_logs, best_vloss, epoch_number

def train(loss_fn, optimizer, model, training_logs, validation_loader, training_loader, EPOCHS, checkpoint_path=None, device='cpu'):
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        training_logs, best_vloss, epoch_number = load_checkpoint(model, optimizer, training_logs, checkpoint_path, device)
    
    t_0_accelerated = time.time()
    for epoch in range(epoch_number, EPOCHS):
        train_loss, train_correct = 0, 0
        model.train(True)

        for i, data in enumerate(training_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).float().sum().item()

        training_logs["train_loss"].append(train_loss / len(training_loader))
        training_logs["train_acc"].append(train_correct / len(training_loader.dataset))

        model.eval()
        valid_loss, valid_correct = 0, 0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
                voutputs = model(vinputs)

                valid_loss += loss_fn(voutputs, vlabels).item()
                valid_correct += (voutputs.argmax(1) == vlabels).float().sum().item()

            training_logs["validate_loss"].append(valid_loss / len(validation_loader))
            training_logs["validate_acc"].append(valid_correct / len(validation_loader.dataset))

        print(f"Epochs {epoch+1}".ljust(10), end='')
        for key in training_logs.keys():
            print(f"{key}: {training_logs[key][-1]:.5f}", end=" ")
        print()
        print("-"*80)

        if checkpoint_path:
            torch.save(model.state_dict(), checkpoint_path + "model.pth")
            torch.save(optimizer.state_dict(), checkpoint_path + "opt.pth")
            torch.save(training_logs, checkpoint_path + 'training_logs.pth')
            if best_vloss > valid_loss:
               torch.save(model.state_dict(), checkpoint_path + "best_model.pth")
               best_vloss = valid_loss

    t_end_accelerated = time.time()-t_0_accelerated
    print(f"Time consumption for accelerated CUDA training (device:{device}): {t_end_accelerated} sec")