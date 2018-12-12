! -*- coding: utf-8 -*-
! vim:fenc=utf-8
!
! ===================================================================
! core.f90
! Copyright (C) 2018 Daniel Santiago <dpelaez@gmail.com>
!
! Distributed under terms of the GNU/GPL license.
!
! ===================================================================

      
      subroutine check_dimensions(W, nfrqs, ntime, npoints, neqs)
!
!     Return the dimensions to work with the array W
!
      implicit none
      complex, dimension(:,:,:), intent(in) :: W
      integer, intent(out) :: nfrqs, ntime, npoints, neqs
        
      nfrqs   = size(W, 1)
      ntime   = size(W, 2)
      npoints = size(W, 3)
      neqs    = int(npoints * (npoints-1) / 2)

      end subroutine



! ===================================================================

      subroutine position_and_phase(W, x, y, XX, Dphi, &
                                    nfrqs, ntime, npoints, neqs)
!
!     This subroutine computes the wavenumber pair (kx, ky) for a
!     given wavelts coefficients and positions. This subroutine is
!     though to be part of the Wavelet Directional Method proposed 
!     by Donelan et al. (1996)
    

!     varible declaration    
      implicit none
      complex*8, intent(in) :: W(nfrqs, ntime, npoints)
      real*8, intent(in)    :: x(ntime, npoints), y(ntime, npoints)
      integer, intent(in)   :: nfrqs, ntime, npoints, neqs
!
      integer :: i, j, k, n, ij
      real*8  :: phase(nfrqs, ntime, npoints)
!
      real*8, intent(out) :: XX(ntime, neqs, 2), Dphi(nfrqs, ntime, neqs)
!

!     compute and phase from the wavelet coefficients
      phase = atan2(-aimag(W), real(W))

      ! loop for each unique pair of points
      !    i,j ---> move along points
      !    k   ---> move along time
      !    n   ---> move along frequencies
      !    ij  ---> move along number of pairs
      ij = 1
      do i = 1, npoints
        !
        do j = i+1, npoints
          !
          ! distances between pairs for each time
          do k = 1, ntime
            XX(k,ij,1) = x(k,j) - x(k,i)
            XX(k,ij,2) = y(k,j) - y(k,i)
            !
            ! difference of phases
            do n = 1, nfrqs
              Dphi(n,k,ij) = phase(n,k,j) - phase(n,k,i)
            end do
          !
          end do
          !
          ! acumulate counter
          ij = ij + 1
        end do
      end do
      
      end subroutine




! ===================================================================

      subroutine compute_wavenumber(XX, Dphi, kx, ky, &
                                    nfrqs, ntime, neqs)

!     apply least squares method to estimate the wavenumber
      
!     variable declaration
      implicit none
!
      real*8, intent(in)  :: XX(ntime, neqs, 2), Dphi(nfrqs, ntime, neqs)
      integer, intent(in) :: nfrqs, ntime, neqs
      real*8, intent(out) :: kx(nfrqs, ntime), ky(nfrqs, ntime)
!
      integer :: i, j, m, n
      real*8 :: detinv
      real*8 :: XTX(2,2), XTX_inv(2,2), XTDphi(2), kk(2)

      ! loop for each frequency
      do i = 1, nfrqs
        !
        ! loop for each time
        do j = 1, ntime
          !
          ! --- least square estimation of vector kk=(kx, ky) ---
          !     the LSR of k is given by:
          !       kk^LS = (X^T X)^-1 X^T Dphi
          !
          ! first term
          XTX(:,:) = matmul(transpose(XX(j,:,:)), XX(j,:,:)) 
          !
          ! inverse of the first term
          detinv = 1 / (XTX(1,1)*XTX(2,2) - XTX(1,2)*XTX(2,1))
          do m = 1, 2
            do n = 1, 2
              XTX_inv(m,n) = (-1)**(m+n) * detinv * XTX(n,m)
            end do
          end do
          !
          ! second term
          XTDphi(:) = matmul(transpose(XX(j,:,:)), Dphi(i,j,:))
          !
          ! compute wavenumber components
          kk(:) = matmul(XTX_inv, XTDphi)
          kx(i,j) = kk(1)
          ky(i,j) = kk(2)
          !
        end do
      end do

      end subroutine



! ===================================================================

      subroutine directional_spreading(W, kx, ky, D, &
                                       nfrqs, ntime, npoints)

!     This function computes the directional spreading function. It is
!     based on finding the 
      
!     variable declaration
      implicit none
!
      complex*8, intent(in) :: W(nfrqs, ntime, npoints)
      real*8, intent(in)    :: kx(nfrqs, ntime), ky(nfrqs, ntime)
      integer, intent(in)   :: nfrqs, ntime, npoints
!
      integer :: i, j, k, ix(ntime)
      real*8 :: kappa(nfrqs, ntime), theta(nfrqs, ntime)
      real*8 :: power(nfrqs, ntime)
      real*8 :: theta_degrees(nfrqs, ntime), weight, m0(nfrqs)
      real*8, parameter :: pi = 4.d0*DATAN(1.d0)
!
      real*8, intent(out) :: D(360, nfrqs)


!     compute power of each pair frequency-time
      power = sum(abs(W)**2, 3) / npoints

!     compute magnitude and direction of the wavenumber vector
      kappa = sqrt(kx**2 + ky**2)
      theta = atan2(ky, kx)

!     round angles to a resolution of 1 degree and correct angle to be
!     measured counterclockwise from east and wrapped from 0 to 360
      theta_degrees = modulo(nint(theta * 180./pi), 360)

!     loop for each frequcency
      do j = 1, nfrqs
        ! 
        ! loop for each direction
        do i = 0, 359
          !
          ! loop for each time
          do k = 1, ntime
            if (theta_degrees(j,k) == i) then
              ix(k) = 1
            else
              ix(k) = 0
            end if
          end do
          !
          ! comute weight
          if (sum(ix) == 0) then
            weight = 0.
          else
            weight = sum(power(j,ix)) / ntime
          end if
          !
          ! compute the directional spreading function
          D(i+1,j) = sum(ix) * weight
          !
        end do
      end do

!     normalize to satisfy int(D) = 1 for each direction
      m0 = sum(D, 1) * pi/180
      do i = 1, 360
        D(i,:) = D(i,:) / m0(:)
      end do

      end subroutine

! --- end of file ---
