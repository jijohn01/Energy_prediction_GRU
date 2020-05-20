from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

import datetime

import matplotlib
import sys
import os


class Training():
    def __init__(self,model):
        self.model = model

    def train(self, args, model, criterion, device, train_loader,test_loader,train_loss_loader,optimizer, epoch,
              writer, normalize_factor, steps, best_loss):
        model.train()
        steps = steps
        best_loss = best_loss
        is_best = True

        for batch_idx, batch_ in enumerate(train_loader):
            model.train()
            batch = batch_[0].clone().detach().to(device=device)
            info = batch_[1].clone().detach().to(device=device)
            before_lim = batch_[2][0].item()
            after_lim = batch_[3][0].item()

            data,target = self.batch_to_in_and_target(args,batch,before_lim,device)

            optimizer.zero_grad()
            output = model(data)

            output_=output[:,-after_lim:,:]
            target_=target[:,-after_lim:,:]
            loss = criterion(output_, target_)
            loss.backward()
            optimizer.step()
            sys.stdout.write('\rTrain Epoch: {} [{}/{} ({:.0f}%)] step : {}\tLoss: {:.16f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), steps, loss.item()))
            steps += 1

            if (steps % args.log_interval == 1) & (steps > args.log_pass):
                train_mse, train_mae, train_won_sum_mae, train_won_mae,train_mae_list,train_var = self.eval(args, train_loss_loader, model,
                                                                                   criterion, device,
                                                                                   normalize_factor,teacher_forcing=True)
                test_mse, test_mae, test_won_sum_mae, test_won_mae,test_mae_list,test_var= self.eval(args, test_loader, model, criterion,
                                                                               device, normalize_factor,teacher_forcing=True)

                self.write_energy2tensorboard(train_mse, test_mse, train_won_mae, steps, writer, 'loss_mse')
                self.write_energy2tensorboard(train_won_sum_mae, test_won_sum_mae, test_won_mae, steps, writer,
                                              'won_mae')
                writer.add_scalars('var',{'train':train_var,'test':test_var},steps)

                for i in range(30):
                    writer.add_scalars('예측 길이에 따른 오차', {'train won'+str(i+1): train_mae_list[i], 'test won'+str(i+1): test_mae_list[i]}, steps)
                try:
                    is_best = test_won_sum_mae < best_loss
                except:
                    best_loss = test_won_sum_mae
                    is_best = test_won_sum_mae < best_loss
                best_loss = min(test_won_sum_mae, best_loss)
                if is_best:
                    self.save_checkpoint(args, {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'steps': steps,
                    }, steps=steps, is_best=is_best)

        self.save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'steps': steps,
        }, steps=steps, is_best=is_best)

        return steps, best_loss

    def gen_limit(self):
        before_lim = np.random.randint(5, 9)
        if before_lim <= 27:  # 31-5 +1 즉 이전거 26번째 것까지 사용한다.
            after_lim = np.random.randint(31 - before_lim + 1, 31)
        else:
            after_lim = np.random.randint(5, 11)
        return before_lim,after_lim

    def batch_to_in_and_target(self,args,batch,before_lim,device):
        if args.isweather:
            edited = batch[:,:,:13].clone().detach()
            edited[:,before_lim,5:]=batch[:,before_lim,5+8:5+(8*2)].clone().detach() #today
            try:
                edited[:, before_lim+1, 5:] = batch[:, before_lim+1, 5 + (8*2) :5 + (8 * 3)].clone().detach()#today +1
                edited[:,before_lim+2,5:] =batch[:,before_lim+2,5+(8*3):5+(8*4)].clone().detach() #today+2
                edited[:,before_lim+3:,5:] =batch[:,before_lim+3:,5+(8*4):].clone().detach()
            except:
                pass
        else:
            edited = batch[:, :, :5].clone().detach()

        inputs = edited[:,:-1,:].clone().detach().to(device=device)
        target = edited[:,1:,2:5].clone().detach().to(device=device)

        return inputs,target

    def eval(self,args,data_loader,model,criterion,device,normalize_factor,teacher_forcing=False):
        #배치마다 로스랑 돈 기록 전체 평균은 따로 변수 설정
        model.eval()
        energy_max = torch.from_numpy(np.asarray(normalize_factor[1][1:4],dtype=float)).cuda()
        energy_min = torch.from_numpy(np.asarray(normalize_factor[2][1:4],dtype=float)).cuda()
        energy_max_min_gap = energy_max-energy_min
        won_mae_list=[]

        with torch.no_grad():
            for idx,batch_ in enumerate(data_loader):
                # print(idx)
                batch = batch_[0].clone().detach().to(device=device)
                info = batch_[1].clone().detach().to(device=device)
                before_lim = batch_[2][0].item()
                after_lim = batch_[3][0].item()
                day_in_month = info[:,-1]
                # if before_lim +after_lim-1 < torch.max(day_in_month):
                #     continue

                data,target = self.batch_to_in_and_target(args,batch,before_lim,device)

                output = model(data, teacher_forcing=False,before_lim=before_lim,after_lim=after_lim)
                output_= output[:,-after_lim:,:]
                target_= target[:,-after_lim:,:]

                try:
                    mse = mse + criterion(output_,target_)
                    mae = mae + F.l1_loss(output_,target_)
                    mae_var = mae_var+torch.var(output_-target_)

                except:
                    mse = criterion(output_,target_)
                    mae = F.l1_loss(output_,target_)
                    mae_var = torch.var(output_-target_)

                # 입력일 + 예측일이 31일이하이면 (31-예측일수) 이전일부터 입력일 직전까지를 타겟에 합침
                if before_lim + after_lim-1 <= 31:
                    front = batch_[-1][:,:-before_lim].cuda()
                    target = torch.cat([front, target], dim=1)

                predicted_nor = torch.sum(output_, dim=1).double()
                target = target.double()

                test_pred = predicted_nor.clone().detach()
                test_target = torch.zeros((batch.__len__(), 3), dtype=torch.float64).cuda()
                #
                for j in range(torch.min(-day_in_month).int(), torch.max(-day_in_month).int() + 1, 1): # j = -31 ~ -28
                    id_tensor = (-day_in_month <= j).double() #day in month에서 j보다 작은 값을 가진 곳 1, 나머지 0
                    test_pred[:, 0] = test_pred[:, 0] + target[:, j, 0] * id_tensor #전기
                    test_pred[:, 1] = test_pred[:, 1] + target[:, j, 1] * id_tensor #수도
                    test_pred[:, 2] = test_pred[:, 2] + target[:, j, 2] * id_tensor #가스
                    test_target[:, 0] = test_target[:, 0] + target[:, j, 0] * id_tensor
                    test_target[:, 1] = test_target[:, 1] + target[:, j, 1] * id_tensor
                    test_target[:, 2] = test_target[:, 2] + target[:, j, 2] * id_tensor
                test_pred += torch.sum(target[:, torch.max(-day_in_month).int() + 1:-after_lim], dim=1)
                test_target += torch.sum(target[:, torch.max(-day_in_month).int() + 1:], dim=1)
                try:
                    use_mae_nor = use_mae_nor + F.l1_loss(test_pred,test_target)
                except:
                    use_mae_nor = F.l1_loss(test_pred, test_target)
                #denormalize
                out_sum = (test_pred * energy_max_min_gap) + (day_in_month.reshape(-1, 1) * energy_min.reshape(-1, 3))
                target_sum = (test_target * energy_max_min_gap) + (
                            day_in_month.reshape(-1, 1) * energy_min.reshape(-1, 3))
                try:
                    use_mae = use_mae + self.MAE(out_sum,target_sum)
                except:
                    use_mae = self.MAE(out_sum, target_sum)
                # 요금으로 변환
                out_sum[:, 0] = self.elec2won_jyb(out_sum[:, 0])
                out_sum[:, 1] = self.water2won_jyb(out_sum[:, 1])
                out_sum[:, 2] = self.gas2won_jyb(out_sum[:, 2])
                target_sum[:, 0] = self.elec2won_jyb(target_sum[:, 0])
                target_sum[:, 1] = self.water2won_jyb(target_sum[:, 1])
                target_sum[:, 2] = self.gas2won_jyb(target_sum[:, 2])

                try:
                    won_sum_mae =won_sum_mae + F.l1_loss(out_sum, target_sum)
                    won_mae = won_mae + self.MAE(out_sum, target_sum)
                    mae_var = mae_var + torch.var(out_sum - target_sum)

                except:
                    won_sum_mae = F.l1_loss(out_sum, target_sum)
                    won_mae = self.MAE(out_sum, target_sum)
                    mae_var = torch.var(out_sum - target_sum)
                # dt5 = datetime.datetime.now()
                # print('검증 5{}'.format(dt5 - dt4))
                won_mae_list.append(F.l1_loss(out_sum, target_sum)*3)

                if idx+1>=30:
                    length = len(won_mae_list)
                    won_mae = won_mae/length
                    won_sum_mae = won_sum_mae/(length)
                    mse = mse/(length)
                    mae = mae/(length)

                    break

        return mse,mae,won_sum_mae*3,won_mae,won_mae_list,mae_var


    def write_energy2tensorboard(self,train_loss, test_loss, err, epoch, writer, name=""):
        writer.add_scalars('Loss' + name, {'train loss': train_loss, 'test loss': test_loss}, epoch)
        writer.add_scalars('energy_prediction_mae' + name, {'elec': err[0], 'water': err[1], 'gas': err[2]}, epoch)

    def save_checkpoint(self, args, state, filename='checkpoint_lstm_std.pth.tar', steps=0, is_best=False):
        if not os.path.isdir('./model/' + args.name + '/best'):
            os.makedirs('./model/' + args.name + '/best')
        torch.save(state, './model/' + args.name + '/' + str(steps) + filename)
        # if is_best:
        #     shutil.copyfile('./model/'+args.name+'/'+str(steps)+filename, './model/'+args.name+'/best/model_best_lstm_std_'+str(steps)+'.pth.tar')

    def MAE(self,output,target):
        err = abs(output-target)
        err =torch.mean(err,dim=0)
        return err

    def elec2won_jyb(self,elec):
        out = elec.clone()
        out[elec <= 200] = 910 + (elec[elec <= 200] * 93.9)
        out[(elec > 200) * (elec <= 400)] = 1600 + (200 * 93.9) + ((elec[(elec > 200) * (elec <= 400)] - 200) * 187.9)
        out[elec > 400] = 7300 + (200 * 93.9) + (200 * 187.8) + ((elec[elec > 400] - 400) * 280.6)
        return out

    def water2won_jyb(self, water):
        numberoffamily = 3
        avg_water = water / numberoffamily
        mm_fair = 3000  # mm:fair 15:1080, 20:3000, 25:5200, 32:9400 : 150mm 아파트 한동에 60가구 정도 살때 1/n 하면 3250원 정도 한다.
        sangsudo = avg_water.clone()
        hasudo = avg_water.clone()
        budam = avg_water.clone()

        con = [avg_water <= 30, (avg_water > 30) * (avg_water <= 50), avg_water > 50]

        sangsudo[con[0]] = mm_fair + (avg_water[con[0]] * 360) * numberoffamily
        hasudo[con[0]] = (avg_water[con[0]] * 400) * numberoffamily
        budam[con[0]] = (avg_water[con[0]] * 170) * numberoffamily

        sangsudo[con[1]] = mm_fair + ((30 * 360) + ((avg_water[con[1]] - 30) * 550)) * numberoffamily
        hasudo[con[1]] = (30 * 400 + (avg_water[con[1]] - 30) * 930) * 30
        budam[con[1]] = ((avg_water[con[1]] - 30) * 170) * numberoffamily

        sangsudo[con[2]] = mm_fair + ((30 * 360) + (20 * 550) + ((avg_water[con[2]] - 50) * 790)) * numberoffamily
        hasudo[con[2]] = ((30 * 400)) + (20 * 930) + ((avg_water[con[2]] - 50) * 1420) * numberoffamily
        budam[con[2]] = ((avg_water[con[2]] - 50) * 170) * numberoffamily

        out = sangsudo + hasudo + budam

        return out

    def gas2won_jyb(self, gas):
        correction_factor = 0.9992
        avg_cal = 42.689
        unit_price = 15.3449
        out = ((gas * correction_factor * avg_cal * unit_price) + 1000) * 1.1  # (사용량 * 보정계수 * 평균열량 * 요금단가 + 기본료)+부가세(10%)
        return out